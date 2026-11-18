"""First-class offline ligand assignment with the local-v5 decision tree."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

from rdkit import Chem

from .analogy import PenaltyModel
from .assignment import AssignedTerm, _lookup_term
from .charges import ChargeModel, ChargeResult, assign_charges, formal_baseline
from .errors import AutoCGENFFError
from .molecule import Kind, Term, inventory, prepare_molecule
from .parameters import ParameterRecord, Parameters, Types, read_parameters
from .topology import Topology, improper_source_centers, read_topology
from .v5 import _type_prepared
from .v5_inputs import SubstitutionEvidence, observed_alkyl_model, read_native_increments

_LINEAR = frozenset({"CG1N1", "CG1T1", "CG1T2"})
# These host types have LONEPAIR declarations in the local v5 RTF. CLGA1
# and BRGA1 also occur without a site: type alone cannot choose the model.
_LP_HOSTS = frozenset({"CLGR1", "BRGR1", "IGR1", "CLGA1", "BRGA1"})


@dataclass(frozen=True, slots=True)
class V5ForceField:
    """Reusable local input bundle; no ligand-specific templates are selected."""

    parameters: Parameters
    topology: Topology
    charges: ChargeModel
    increments_path: Path
    penalties: PenaltyModel
    substitution_evidence: tuple[SubstitutionEvidence, ...]
    improper_centers: Mapping[Types, int]

    @classmethod
    def from_directory(
        cls, directory: Path, *, increments_path: Path | None = None
    ) -> "V5ForceField":
        """Load caller's local cgenff PRM, RTF and annotated v5 increment file.

        An explicit increments_path takes precedence over the adjacent conventional
        filename. Missing or malformed files raise AutoCGENFFError.
        """
        directory = Path(directory).resolve()
        path = (
            Path(increments_path).resolve()
            if increments_path is not None
            else directory / "par_all36_cgenff_increments.prm"
        )
        try:
            charges = read_native_increments(path)
            parameters = read_parameters(directory / "par_all36_cgenff.prm")
            topology = read_topology(directory / "top_all36_cgenff.rtf")
            penalties, evidence = observed_alkyl_model(parameters)
        except (OSError, UnicodeError) as exc:
            raise AutoCGENFFError(f"Cannot load local v5 force field: {exc}") from exc
        return cls(
            parameters,
            topology,
            charges,
            path,
            penalties,
            evidence,
            improper_source_centers(topology),
        )


@dataclass(frozen=True, slots=True)
class V5AssignedAtom:
    """One real atom in the input's explicit-H order; unresolved charges are None."""

    index: int
    element: str
    name: str
    atom_type: str
    formal_charge: int
    rule_charge: int
    improper_center: bool
    trace: tuple[str, ...]
    warnings: tuple[str, ...]
    mass: float | None
    partial_charge: float | None
    charge_penalty: float | None


@dataclass(frozen=True, slots=True)
class V5Bond:
    """Covalent input edge in the very Kekulé graph used for rule evaluation."""

    atoms: tuple[int, int]
    order: int


@dataclass(frozen=True, slots=True)
class V5Assignment:
    """Real atoms, required mechanical inventory, omissions and independent charge paths."""

    atoms: tuple[V5AssignedAtom, ...]
    bonds: tuple[V5Bond, ...]
    terms: tuple[AssignedTerm, ...]
    omitted_linear_propers: tuple[AssignedTerm, ...]
    charge_diagnostics: ChargeResult
    nbfix: tuple[ParameterRecord, ...]
    parameter_source: str
    topology_source: str
    charge_source: str
    issues: tuple[str, ...]
    substitution_profile: str = "local-v1/observed-acyclic-alkyl (provisional)"

    @property
    def is_complete(self) -> bool:
        """Assigned relative to these local models; not a quality guarantee."""
        return not self.issues and all(term.match is not None for term in self.terms)


def _resolve_improper(
    center: int, neighbors: tuple[int, ...], types: Types, ff: V5ForceField
) -> AssignedTerm:
    """Try center-first permutations; prefer exact, then wildcard, then analogy.

    Equal type sequences differing only by like-typed neighbors use index order.
    Different equally good sequences are not resolved by an arbitrary sort.
    """
    candidates = {}
    for outer in permutations(sorted(neighbors)):
        path = (center, *outer)
        key = tuple(types[i] for i in path)
        if key in candidates or ff.improper_centers.get(key) == 3:
            continue
        status, penalty, match = _lookup_term(
            ff.parameters, Kind.IMPROPER, key, ff.penalties, 0, ff.improper_centers
        )
        if match is not None:
            candidates[key] = AssignedTerm(Term(Kind.IMPROPER, path), key, status, penalty, match)
    default_path = (center, *sorted(neighbors))
    key = tuple(types[i] for i in default_path)
    default_term = Term(Kind.IMPROPER, default_path)
    if not candidates:
        return AssignedTerm(default_term, key, "missing")
    ranks = {"exact": 0, "wildcard": 1, "analogy": 2}

    def rank(t: AssignedTerm) -> tuple[int, float]:
        return ranks[t.status], t.penalty or 0.0

    best_rank = min(map(rank, candidates.values()))
    best = [t for t in candidates.values() if rank(t) == best_rank]
    if len(best) > 1:
        return AssignedTerm(default_term, key, "ambiguous_improper_order")
    return best[0]


def _atom_names(molecule: Chem.Mol) -> list[str]:
    """Keep declared atom names. Generate element/ordinal names only for unnamed atoms."""
    names = []
    for atom in molecule.GetAtoms():
        declared = {
            atom.GetProp(key).strip()
            for key in ("_TriposAtomName", "molFileAlias", "atomLabel", "_Name", "name")
            if atom.HasProp(key)
        }
        info = atom.GetPDBResidueInfo()
        if info is not None:
            declared.add(info.GetName().strip())
        declared.discard("")
        if len(declared) > 1:
            raise AutoCGENFFError(f"Conflicting input atom names at index {atom.GetIdx()}")
        names.append(declared.pop() if declared else "")
    used = {name.upper() for name in names if name}
    counts = Counter()
    for atom, name in zip(molecule.GetAtoms(), names, strict=True):
        if not name:
            element = atom.GetSymbol().upper()
            while not name or name in used:
                counts[element] += 1
                name = f"{element}{counts[element]}"
            names[atom.GetIdx()] = name
            used.add(name)
    return names


def parameterize_v5(
    source: str | Path | Chem.Mol,
    force_field: V5ForceField | Path,
    *,
    increments_path: Path | None = None,
) -> V5Assignment:
    """Type and assign a ligand with native local v5 force-field files, without a template.

    Pass a directory (and optional external increments_path), or reuse a loaded
    V5ForceField. Inspect is_complete and issues before use. Malformed inputs,
    unsupported typing or inconsistent rule charge seeds raise AutoCGENFFError.
    """
    molecule = prepare_molecule(source)
    typed = _type_prepared(molecule)
    if not typed.is_complete:
        details = [
            f"{a.index}:{a.element} ({', '.join(a.errors)})"
            for a in typed.atoms
            if a.atom_type is None
        ]
        raise AutoCGENFFError(f"No usable local-v5 type: {', '.join(details)}")
    types = tuple(a.atom_type for a in typed.atoms if a.atom_type is not None)
    seeds = formal_baseline(molecule, tuple(a.rule_charge for a in typed.atoms))
    if isinstance(force_field, V5ForceField):
        if increments_path is not None:
            raise AutoCGENFFError("increments_path cannot override a loaded V5ForceField")
        ff = force_field
    else:
        ff = V5ForceField.from_directory(force_field, increments_path=increments_path)
    issues = []
    real_paths = inventory(molecule)
    required, omitted = [], []
    cache = {}
    for term in (*real_paths, *(Term(Kind.NONBONDED, (i,)) for i in range(len(types)))):
        key = tuple(types[i] for i in term.atoms)
        if term.kind == Kind.PROPER and _LINEAR.intersection(key[1:3]):
            native = ff.parameters.match(term.kind, key)
            if native is None or native.wildcard:
                omitted.append(AssignedTerm(term, key, "undefined_linear_proper"))
                continue
        lookup = (term.kind, key)
        if lookup not in cache:
            cache[lookup] = _lookup_term(ff.parameters, term.kind, key, ff.penalties, None)
        required.append(AssignedTerm(term, key, *cache[lookup]))
    for atom in typed.atoms:
        if atom.improper_center:
            neighbors = tuple(
                n.GetIdx() for n in molecule.GetAtomWithIdx(atom.index).GetNeighbors()
            )
            if len(neighbors) == 3:
                required.append(_resolve_improper(atom.index, neighbors, types, ff))
            else:
                issues.append(
                    f"Rule-marked improper center {atom.index} has {len(neighbors)} neighbors"
                )
    for term in required:
        if term.match is None:
            issues.append(f"{term.status}: {term.term.kind} {term.term.atoms} {term.target_types}")
    if ff.parameters.unsupported_sections:
        issues.append(f"Unsupported PRM sections: {', '.join(ff.parameters.unsupported_sections)}")
    # Never delete charge paths merely because a mechanical torsion is undefined.
    charges = assign_charges(
        molecule, types, real_paths, ff.charges, penalties=ff.penalties, baseline=seeds
    )
    if charges.charges is None:
        issues.append("Unresolved native charge increments (see charge_diagnostics.terms)")
    lp_hosts = tuple(i for i, label in enumerate(types) if label in _LP_HOSTS)
    if lp_hosts:
        issues.append(
            f"Unresolved halogen lone-pair representation at {lp_hosts}; cannot export real-atom charges as site charges"
        )
    atoms = []
    names = _atom_names(molecule)
    for atom, label in zip(typed.atoms, types, strict=True):
        mass = ff.topology.masses.get(label)
        if (
            mass is None
            or mass <= 0
            or ff.topology.elements.get(label, atom.element) != atom.element
        ):
            issues.append(f"Missing or inconsistent RTF MASS for atom {atom.index} ({label})")
        if molecule.GetAtomWithIdx(atom.index).GetIsotope():
            issues.append(f"Explicit isotope at atom {atom.index} not represented by the base MASS")
        atoms.append(
            V5AssignedAtom(
                atom.index,
                atom.element,
                names[atom.index],
                label,
                atom.formal_charge,
                atom.rule_charge,
                atom.improper_center,
                atom.trace,
                atom.warnings,
                mass,
                None if lp_hosts or charges.charges is None else charges.charges[atom.index],
                None
                if lp_hosts or charges.atom_penalties is None
                else charges.atom_penalties[atom.index],
            )
        )
    bonds = []
    for term in real_paths:
        if term.kind == Kind.BOND:
            i, j = term.atoms
            bond = molecule.GetBondBetweenAtoms(i, j)
            bonds.append(V5Bond((i, j), int(bond.GetBondTypeAsDouble())))
    return V5Assignment(
        tuple(atoms),
        tuple(bonds),
        tuple(required),
        tuple(omitted),
        charges,
        ff.parameters.nbfix_for(types),
        str(ff.parameters.source),
        str(ff.topology.source),
        str(ff.increments_path),
        tuple(issues),
    )
