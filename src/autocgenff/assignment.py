"""End-to-end parameter and charge assignment with explicit provenance."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from .analogy import PenaltyModel, best_analogy
from .charges import ChargeModel, ChargeResult, assign_charges
from .errors import AutoCGENFFError
from .molecule import Kind, Term, inventory, prepare_molecule
from .parameters import ParameterMatch, ParameterRecord, Parameters, Types
from .topology import TemplateMatch, Topology, improper_source_centers, match_template
from .typing import TypedAtom, TypingRule, type_atoms


@dataclass(frozen=True, slots=True)
class AssignmentOptions:
    """Independent typing/charge options, or an explicitly selected RTF residue."""

    rules: tuple[TypingRule, ...] = ()
    include_builtin: bool = True
    penalties: PenaltyModel | None = None
    charge_model: ChargeModel | None = None
    formal_baseline: tuple[float, ...] | None = None
    impropers: tuple[tuple[int, ...], ...] | None = None
    topology: Topology | None = None
    template: str | None = None
    template_smiles: str | None = None
    atom_map: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class AtomAssignment:
    """Atom type and optional assigned charge, in input explicit-H order."""

    index: int
    element: str
    formal_charge: int
    atom_type: str
    typing_source: str
    partial_charge: float | None
    charge_penalty: float | None
    template_name: str | None


@dataclass(frozen=True, slots=True)
class AssignedTerm:
    """Parameter records and lookup provenance for a bonded term or LJ site."""

    term: Term
    target_types: Types
    status: str
    penalty: float | None = None
    match: ParameterMatch | None = None


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    """Diagnostic-only result; never a guarantee of simulation-ready parameters."""

    atoms: tuple[AtomAssignment, ...]
    terms: tuple[AssignedTerm, ...]
    nbfix: tuple[ParameterRecord, ...]
    parameter_source: str
    template_source: TemplateMatch | None
    charge_source: str | None
    charge_diagnostics: ChargeResult | None
    issues: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """Whether this diagnostic has all requested terms and charges."""
        return not self.issues and all(term.match is not None for term in self.terms)


def parameterize(
    source: str | Path | Chem.Mol, parameters: Parameters, options: AssignmentOptions | None = None
) -> AssignmentResult:
    """Prepare, type and assign explicit terms without inventing missing parameters.

    Unsupported typing, invalid input and invalid/mismatched templates raise
    AutoCGENFFError. Missing charges or parameters are returned as diagnostics.
    """
    settings = options if options is not None else AssignmentOptions()
    molecule = prepare_molecule(source)
    template = _select_template(molecule, settings)
    if template is None:
        typings = type_atoms(molecule, settings.rules, include_builtin=settings.include_builtin)
        bonded = inventory(molecule, settings.impropers or ())
    else:
        typings = tuple(
            TypedAtom(atom.atom_type, f"RTF:{template.name}") for atom in template.atoms
        )
        bonded = inventory(
            molecule,
            template.impropers,
            explicit_angles=template.angles,
            explicit_propers=template.propers,
            auto_angles=template.auto_angles,
            auto_propers=template.auto_propers,
        )
    types = tuple(atom.atom_type for atom in typings)
    terms = _assign_terms(
        molecule, bonded, types, parameters, settings.penalties, settings.topology
    )
    charge_result = None
    charges = scores = None
    charge_source = None
    issues = []
    if template is not None:
        charges = tuple(atom.charge for atom in template.atoms)
        charge_source = "named_rtf_template"
    elif settings.charge_model is not None:
        charge_result = assign_charges(
            molecule,
            types,
            bonded,
            settings.charge_model,
            penalties=settings.penalties,
            baseline=settings.formal_baseline,
        )
        charges, scores = charge_result.charges, charge_result.atom_penalties
        if charges is not None:
            charge_source = "caller_increment_model"
    if charges is None:
        issues.append(
            "Missing charge increments"
            if charge_result is not None
            else "No charge model or template supplied"
        )
    if template is None and settings.impropers is None:
        issues.append(
            "Improper dependencies unverified; supply explicit terms or assert none are needed"
        )
    if parameters.unsupported_sections:
        issues.append(f"Unsupported PRM sections: {', '.join(parameters.unsupported_sections)}")
    atoms = tuple(
        AtomAssignment(
            index,
            atom.GetSymbol(),
            atom.GetFormalCharge(),
            typings[index].atom_type,
            typings[index].source,
            None if charges is None else charges[index],
            None if scores is None else scores[index],
            None if template is None else template.atoms[index].name,
        )
        for index, atom in enumerate(molecule.GetAtoms())
    )
    return AssignmentResult(
        atoms,
        terms,
        parameters.nbfix_for(types),
        str(parameters.source),
        template,
        charge_source,
        charge_result,
        tuple(issues),
    )


def _select_template(molecule: Chem.Mol, settings: AssignmentOptions) -> TemplateMatch | None:
    if settings.template is None:
        if settings.template_smiles is not None or settings.atom_map is not None:
            raise AutoCGENFFError("Template identity or atom map requires a named template")
        if settings.formal_baseline is not None and settings.charge_model is None:
            raise AutoCGENFFError("Formal baseline requires an increment model")
        return None
    if settings.topology is None:
        raise AutoCGENFFError("Named template requires an RTF topology")
    if (
        settings.rules
        or settings.charge_model is not None
        or settings.impropers is not None
        or settings.formal_baseline is not None
    ):
        raise AutoCGENFFError("Named-template typing, charges and impropers cannot be overridden")
    return match_template(
        molecule,
        settings.topology,
        settings.template,
        atom_map=settings.atom_map,
        reference_smiles=settings.template_smiles,
    )


def _assign_terms(
    molecule: Chem.Mol,
    bonded: Sequence[Term],
    types: Types,
    parameters: Parameters,
    penalties: PenaltyModel | None,
    topology: Topology | None,
) -> tuple[AssignedTerm, ...]:
    assignments = []
    cache = {}
    source_centers = (
        improper_source_centers(topology)
        if topology is not None
        and penalties is not None
        and any(term.kind == Kind.IMPROPER for term in bonded)
        else {}
    )
    for term in (*bonded, *(Term(Kind.NONBONDED, (index,)) for index in range(len(types)))):
        key = tuple(types[index] for index in term.atoms)
        center = _improper_center(molecule, term) if term.kind == Kind.IMPROPER else None
        lookup = (term.kind, key, center)
        if lookup not in cache:
            cache[lookup] = _lookup_term(
                parameters, term.kind, key, penalties, center, source_centers
            )
        status, score, match = cache[lookup]
        assignments.append(AssignedTerm(term, key, status, score, match))
    return tuple(assignments)


def _improper_center(molecule: Chem.Mol, term: Term) -> int | None:
    centers = [
        position
        for position in (0, 3)
        if all(
            molecule.GetBondBetweenAtoms(term.atoms[position], other) is not None
            for other in term.atoms
            if other != term.atoms[position]
        )
    ]
    return centers[0] if len(centers) == 1 else None


def _lookup_term(
    parameters: Parameters,
    kind: Kind,
    key: Types,
    penalties: PenaltyModel | None,
    center: int | None,
    source_centers: Mapping[Types, int] | None = None,
) -> tuple[str, float | None, ParameterMatch | None]:
    match = parameters.match(kind, key)
    if match is not None:
        return "wildcard" if match.wildcard else "exact", 0.0, match
    # An LJ substitution may change NBFIX and 1-4 behavior. Never do it implicitly.
    if penalties is None or kind == Kind.NONBONDED:
        return "missing", None, None
    if kind == Kind.IMPROPER and (center is None or not source_centers):
        return "missing", None, None
    candidates = (records[0].types for records in parameters.iter_groups(kind))
    analogy = best_analogy(
        kind,
        key,
        candidates,
        penalties,
        improper_center=center,
        source_improper_centers=source_centers,
    )
    if analogy is None:
        return "missing", None, None
    match = parameters.match(kind, analogy.types)
    if match is None:
        raise RuntimeError("Selected analogy does not exist in the parameter database")
    flags = tuple(reverse != analogy.reversed for reverse in match.record_reversed)
    return "analogy", analogy.penalty, ParameterMatch(match.records, flags)
