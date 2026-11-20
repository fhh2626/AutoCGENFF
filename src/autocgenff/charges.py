"""Antisymmetric bond, angle and proper-torsion charge increments."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import fsum, isclose, isfinite, sqrt
from types import MappingProxyType

from rdkit import Chem

from .analogy import PenaltyModel, best_analogy
from .errors import AutoCGENFFError
from .molecule import Kind, Term, _valid_indices
from .parameters import Types

PATH_ORDER = {Kind.BOND: 2, Kind.ANGLE: 3, Kind.PROPER: 4}
ZERO_WEIGHT = 0.05**6
CHARGE_TOLERANCE = 1e-6


def reverse_increments(values: tuple[float, ...]) -> tuple[float, ...]:
    """Reverse an oriented path: reverse its edge increments and negate them."""
    return tuple(-value for value in reversed(values))


@dataclass(frozen=True, slots=True)
class ChargeModel:
    """Explicit increment table in elementary charges; no implicit fitted data.

    Keys are ordered type tuples of length 2, 3 or 4. Entries give one transfer
    from each path atom to the next. A lower max_order selects a reduced model
    explicitly; a missing entry within that order makes per-atom charges unknown.
    The torsion cutoff must be finite and positive. A proper path without an exact
    entry transfers nothing if one of its two central types is in linear_centers.
    """

    increments: Mapping[Types, tuple[float, ...] | None]  # None is explicit native N/A, not missing
    max_order: int = 4
    torsion_cutoff: float = 50.0
    # Canonical key -> original oriented native row and its line number.
    source_rows: Mapping[Types, tuple[Types, int]] = field(default_factory=dict)
    infer_identical_bonds: bool = True
    linear_centers: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.max_order not in (2, 3, 4):
            raise AutoCGENFFError("Increment model max_order must be 2, 3 or 4")
        if not isfinite(self.torsion_cutoff) or self.torsion_cutoff <= 0:
            raise AutoCGENFFError("Torsion cutoff must be finite and positive")
        canonical = {}
        for types, values in self.increments.items():
            values = None if values is None else tuple(values)
            if (
                not isinstance(types, tuple)
                or not 2 <= len(types) <= self.max_order
                or (
                    values is not None
                    and (len(values) != len(types) - 1 or not all(map(isfinite, values)))
                )
                or not all(isinstance(item, str) and item and item != "X" for item in types)
            ):
                raise AutoCGENFFError(f"Invalid increment entry for {types}")
            key = min(types, types[::-1])
            if values is not None and types == types[::-1] and values != reverse_increments(values):
                raise AutoCGENFFError(
                    f"Palindromic increment entry must be reversal-invariant: {types}"
                )
            normal = values if values is None or key == types else reverse_increments(values)
            if canonical.setdefault(key, normal) != normal:
                raise AutoCGENFFError(f"Conflicting reverse increment entries for {types}")
        object.__setattr__(self, "increments", MappingProxyType(canonical))
        object.__setattr__(self, "source_rows", MappingProxyType(dict(self.source_rows)))
        if isinstance(self.linear_centers, str):
            raise AutoCGENFFError("Linear centers must be a collection of explicit atom types")
        centers = frozenset(self.linear_centers)
        if not all(isinstance(item, str) and item and item != "X" for item in centers):
            raise AutoCGENFFError("Linear centers must be explicit atom types")
        object.__setattr__(self, "linear_centers", centers)


@dataclass(frozen=True, slots=True)
class IncrementMatch:
    """Charge diagnostic for a path, including the applied values and provenance."""

    term: Term
    target_types: Types
    source_types: Types | None
    values: tuple[float, ...] | None
    penalty: float | None
    status: str
    reversed: bool = False
    source_line: int | None = None


@dataclass(frozen=True, slots=True)
class ChargeResult:
    """Complete charges or None, with path-level missing diagnostics either way."""

    charges: tuple[float, ...] | None
    atom_penalties: tuple[float, ...] | None
    terms: tuple[IncrementMatch, ...]


def formal_baseline(
    molecule: Chem.Mol, baseline: Sequence[float] | None = None
) -> tuple[float, ...]:
    """Get RDKit formal charges or validate a custom resonance-symmetric baseline."""
    formal = tuple(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    if baseline is None:
        return formal
    values = tuple(baseline)
    if len(values) != len(formal) or not all(isfinite(value) for value in values):
        raise AutoCGENFFError("Formal baseline must have one finite value per explicit atom")
    _check_charge_totals(
        molecule, values, formal, "Custom baseline does not preserve formal charge"
    )
    return values


def _check_charge_totals(
    molecule: Chem.Mol, values: Sequence[float], reference: Sequence[float], message: str
) -> None:
    for fragment in Chem.GetMolFrags(molecule):
        total = fsum(values[i] for i in fragment)
        expected = fsum(reference[i] for i in fragment)
        if not isclose(total, expected, rel_tol=0, abs_tol=CHARGE_TOLERANCE):
            raise AutoCGENFFError(
                f"{message}: component starting at atom {fragment[0]} sums to {total}, expected {expected}"
            )


def assign_charges(
    molecule: Chem.Mol,
    types: Types,
    terms: Sequence[Term],
    model: ChargeModel,
    *,
    penalties: PenaltyModel | None = None,
    baseline: Sequence[float] | None = None,
) -> ChargeResult:
    """Apply increments to valid covalent paths; never return partial charges."""
    count = molecule.GetNumAtoms()
    if len(types) != count:
        raise AutoCGENFFError("Expected one type per explicit atom")
    start = formal_baseline(molecule, baseline)
    deltas = [[] for _ in start]
    errors = [[] for _ in start]
    matches = []
    missing = False
    looked_up = {}
    for term in terms:
        order = PATH_ORDER.get(term.kind)
        if order is None or order > model.max_order:
            continue
        if not _valid_indices(term.atoms, order, count) or any(
            molecule.GetBondBetweenAtoms(left, right) is None
            for left, right in zip(term.atoms, term.atoms[1:], strict=False)
        ):
            raise AutoCGENFFError(f"Invalid {term.kind} charge path: {term.atoms}")
        requested = tuple(types[i] for i in term.atoms)
        key = (term.kind, requested)
        if key not in looked_up:
            looked_up[key] = _match_increment(term.kind, requested, model, penalties)
        source, values, score, status, reverse = looked_up[key]
        row = model.source_rows.get(source) if source is not None else None
        if row is not None:
            reverse = reverse != (row[0] != source)
            source = row[0]
        matches.append(
            IncrementMatch(
                term,
                requested,
                source,
                values,
                score,
                status,
                reverse,
                None if row is None else row[1],
            )
        )
        if status in {"not_applicable", "analogy_not_applicable", "linear_not_applicable"}:
            continue
        if values is None or score is None:
            missing = True
            continue
        # Eq. 4 counts one net transfer per atom *per parameter instance*,
        # not two independent errors on an inner atom of the same instance.
        extended = (0.0, *values, 0.0)
        for k, atom in enumerate(term.atoms):
            net = extended[k] - extended[k + 1]
            deltas[atom].append(net)
            errors[atom].append((abs(net) + ZERO_WEIGHT) ** (1 / 3) * score**2)
    if missing:
        return ChargeResult(None, None, tuple(matches))
    charges = tuple(fsum((value, *delta)) for value, delta in zip(start, deltas, strict=True))
    _check_charge_totals(
        molecule, charges, start, "Charge increments failed to conserve formal charge"
    )
    return ChargeResult(charges, tuple(sqrt(fsum(values)) for values in errors), tuple(matches))


def _match_increment(
    kind: Kind, requested: Types, model: ChargeModel, penalties: PenaltyModel | None
) -> tuple[Types | None, tuple[float, ...] | None, float | None, str, bool]:
    target = min(requested, requested[::-1])
    target_reversed = requested != target
    if target in model.increments:
        exact = model.increments[target]
        if exact is None:
            return target, None, 0.0, "not_applicable", target_reversed
        values = reverse_increments(exact) if target_reversed else exact
        return target, values, 0.0, "exact", target_reversed
    if model.infer_identical_bonds and kind == Kind.BOND and target == target[::-1]:
        return target, (0.0,), 0.0, "identical_bond_types", False
    # A torsion about a linear angle is undefined; the model gives such a path no transfer.
    if kind == Kind.PROPER and model.linear_centers.intersection(target[1:3]):
        return None, None, 0.0, "linear_not_applicable", False
    if penalties is None:
        return None, None, None, "missing", False
    candidates = (
        key
        for key, values in model.increments.items()
        if target != target[::-1] or values is None or values == reverse_increments(values)
    )
    candidate = best_analogy(kind, target, candidates, penalties, for_charge=True)
    if candidate is None:
        return None, None, None, "missing", False
    reverse = candidate.reversed != target_reversed
    if kind == Kind.PROPER and candidate.penalty > model.torsion_cutoff:
        return candidate.types, (0.0, 0.0, 0.0), model.torsion_cutoff, "cutoff", reverse
    values = model.increments[candidate.types]
    if values is None:
        return candidate.types, None, candidate.penalty, "analogy_not_applicable", reverse
    if reverse:
        values = reverse_increments(values)
    return candidate.types, values, candidate.penalty, "analogy", reverse
