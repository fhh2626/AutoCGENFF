"""First-class offline ligand assignment with the local-v5 decision tree."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations
from math import isclose, remainder
from pathlib import Path

from rdkit import Chem

from .analogy import PenaltyModel
from .assignment import AssignedTerm, _lookup_term
from .charges import ChargeModel, ChargeResult, assign_charges, formal_baseline
from .errors import AutoCGENFFError
from .molecule import Kind, Term, inventory
from .parameters import ParameterRecord, Parameters, Types, read_parameters
from .topology import Topology, improper_source_centers, read_topology
from .v5 import _prepare_v5, _type_prepared
from .v5_inputs import (
    LINEAR_CENTERS,
    SubstitutionEvidence,
    observed_v5_penalty_model,
    read_native_increments,
)

# CGenFF 5.0 collinear sigma-hole lone pairs for aromatic halogens
# (Gutierrez et al., J. Chem. Theory Comput. 2016, 12, 4633; top_all36_cgenff.rtf).
_HALOGEN_LONE_PAIR_DISTANCES: Mapping[str, float] = {
    "CLGR1": 1.64,
    "BRGR1": 1.89,
    "IGR1": 2.20,
}
_HALOGEN_LONE_PAIR_TYPE = "LPH"
_HALOGEN_LONE_PAIR_CHARGE = 0.05


def bundled_force_field_dir() -> Path:
    """Return the directory containing the bundled CGenFF 5.0 force-field files."""
    directory = Path(__file__).resolve().parent / "force_fields"
    if not directory.is_dir():
        raise AutoCGENFFError(f"Bundled force-field directory not found: {directory}")
    return directory


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
        cls, directory: Path | None = None, *, increments_path: Path | None = None
    ) -> "V5ForceField":
        """Load cgenff PRM, RTF and annotated v5 increment files.

        Uses the bundled CGenFF 5.0 files when directory is omitted. An explicit
        increments_path takes precedence over the adjacent conventional filename.
        Missing or malformed files raise AutoCGENFFError.
        """
        try:
            if isinstance(directory, str) and not directory.strip():
                raise AutoCGENFFError("Force-field directory path cannot be empty")
            if isinstance(increments_path, str) and not increments_path.strip():
                raise AutoCGENFFError("Increments path cannot be empty")
            base_dir = (
                bundled_force_field_dir() if directory is None else Path(directory).resolve()
            )
            path = (
                Path(increments_path).resolve()
                if increments_path is not None
                else base_dir / "par_all36_cgenff_increments.prm"
            )
            charges = read_native_increments(path)
            parameters = read_parameters(base_dir / "par_all36_cgenff.prm")
            topology = read_topology(base_dir / "top_all36_cgenff.rtf")
            penalties, evidence = observed_v5_penalty_model(parameters, topology)
        except (TypeError, OSError, UnicodeError) as exc:
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
class LonePair:
    """Collinear virtual lone-pair site attached to an aromatic halogen atom."""

    name: str
    atom_type: str
    host: int
    anchor: int
    distance: float
    charge: float
    scale: float = 0.0


@dataclass(frozen=True, slots=True)
class V5Assignment:
    """Real atoms, virtual lone pairs, required mechanical inventory and charge paths."""

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
    substitution_profile: str = "local-v2/observed-cgenff-v5 (provisional)"
    lone_pairs: tuple[LonePair, ...] = ()

    @property
    def is_complete(self) -> bool:
        """Assigned relative to these local models; not a quality guarantee."""
        return not self.issues and all(term.match is not None for term in self.terms)


def _is_even_improper(term: AssignedTerm) -> bool:
    """True when negating the signed improper angle psi leaves energy unchanged."""
    if term.match is None:
        return False
    for record in term.match.records:
        _, multiplicity, psi0 = record.values
        if multiplicity == 0:
            if not isclose(psi0, 0.0, rel_tol=0.0, abs_tol=1e-9):
                return False
        # A periodic improper uses cos(n * psi - phase), not cos(n * (psi - phase)).
        elif not isclose(remainder(psi0, 180.0), 0.0, rel_tol=0.0, abs_tol=1e-9):
            return False
    return True


def _resolve_improper(
    center: int, neighbors: tuple[int, ...], types: Types, ff: V5ForceField
) -> AssignedTerm:
    """Try center-first permutations; prefer exact, then wildcard, then analogy.

    Equal type sequences differing only by like-typed neighbors use index order.
    Competing potentials remain ambiguous. Identical parameters can be resolved
    by topology evidence, the native forward order, or an energy-invariant J/K swap.
    """
    candidates = {}
    seen = set()
    for outer in permutations(sorted(neighbors)):
        path = (center, *outer)
        key = tuple(types[i] for i in path)
        if key in seen or ff.improper_centers.get(key) == 3:
            continue
        seen.add(key)
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
    if len(best) == 1:
        return best[0]
    values = {tuple(row.values for row in t.match.records) for t in best}
    if len(values) == 1:
        declared = [t for t in best if ff.improper_centers.get(t.target_types) == 0]
        if len(declared) == 1:
            return declared[0]
        if all(_is_even_improper(t) for t in best):
            if len({t.target_types[3] for t in best}) == 1:
                return min(best, key=lambda t: t.target_types)
            # For equivalent oxygens, native RTF and PRM can give both a forward
            # center-first order and a reverse-listed order with the same potential.
            forward = [
                t for t in declared
                if t.status == "exact" and not any(t.match.record_reversed)
            ]
            if len(declared) == len(best) and len(forward) == 1:
                return forward[0]
    return AssignedTerm(default_term, key, "ambiguous_improper_order")


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
    force_field: V5ForceField | Path | None = None,
    *,
    increments_path: Path | None = None,
) -> V5Assignment:
    """Type and assign a ligand with native local v5 force-field files, without a template.

    Uses the bundled CGenFF 5.0 files when force_field is omitted, or pass a
    directory (and optional external increments_path), or reuse a loaded
    V5ForceField. Inspect is_complete, issues and lone_pairs before use. Malformed
    inputs, unsupported typing or inconsistent rule charge seeds raise
    AutoCGENFFError. As in type_v5, localized amide and ester pi pairs are
    normalized to their neutral carbonyl contributor. Rule charge seeds must
    still sum to the formal charge of each connected component before parameter
    lookup. Aromatic halogens (CLGR1, BRGR1, IGR1) receive collinear LPH virtual
    lone-pair sites when LPH is defined in the supplied base files.
    """
    molecule = _prepare_v5(source)
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
        if term.kind == Kind.PROPER and LINEAR_CENTERS.intersection(key[1:3]):
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
    names = _atom_names(molecule)
    used_names = {name.upper() for name in names}
    lone_pairs: list[LonePair] = []
    lp_charge_shift: dict[int, float] = {}
    lp_error = False
    lp_index = 1
    for i, label in enumerate(types):
        distance = _HALOGEN_LONE_PAIR_DISTANCES.get(label)
        if distance is None:
            continue
        neighbors = tuple(n.GetIdx() for n in molecule.GetAtomWithIdx(i).GetNeighbors())
        if len(neighbors) != 1:
            issues.append(
                f"Halogen lone-pair host atom {i} ({label}) requires 1 neighbor, found {len(neighbors)}"
            )
            lp_error = True
            continue
        lph_mass = ff.topology.masses.get(_HALOGEN_LONE_PAIR_TYPE)
        lph_nb = ff.parameters.match(Kind.NONBONDED, (_HALOGEN_LONE_PAIR_TYPE,))
        if lph_mass is None or lph_mass != 0.0 or lph_nb is None:
            issues.append(
                f"Missing {_HALOGEN_LONE_PAIR_TYPE} mass or nonbonded entry for halogen lone-pair site at atom {i} ({label})"
            )
            lp_error = True
            continue
        while f"LP{lp_index}" in used_names:
            lp_index += 1
        lp_name = f"LP{lp_index}"
        lp_index += 1
        used_names.add(lp_name)
        lone_pairs.append(
            LonePair(
                name=lp_name,
                atom_type=_HALOGEN_LONE_PAIR_TYPE,
                host=i,
                anchor=neighbors[0],
                distance=distance,
                charge=_HALOGEN_LONE_PAIR_CHARGE,
                scale=0.0,
            )
        )
        lp_charge_shift[i] = lp_charge_shift.get(i, 0.0) + _HALOGEN_LONE_PAIR_CHARGE
    atoms = []
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
        site_charge = (
            None
            if lp_error or charges.charges is None
            else charges.charges[atom.index] - lp_charge_shift.get(atom.index, 0.0)
        )
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
                site_charge,
                None
                if lp_error or charges.atom_penalties is None
                else charges.atom_penalties[atom.index],
            )
        )
    bonds = []
    for term in real_paths:
        if term.kind == Kind.BOND:
            i, j = term.atoms
            bond = molecule.GetBondBetweenAtoms(i, j)
            bonds.append(V5Bond((i, j), int(bond.GetBondTypeAsDouble())))
    all_types = (*types, *(lp.atom_type for lp in lone_pairs))
    return V5Assignment(
        tuple(atoms),
        tuple(bonds),
        tuple(required),
        tuple(omitted),
        charges,
        ff.parameters.nbfix_for(all_types),
        str(ff.parameters.source),
        str(ff.topology.source),
        str(ff.increments_path),
        tuple(issues),
        lone_pairs=tuple(lone_pairs),
    )
