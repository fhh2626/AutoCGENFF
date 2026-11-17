"""Explicit opt-in directional penalties for transferable parameter candidates."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import fsum, inf, isfinite
from types import MappingProxyType

from .errors import AutoCGENFFError
from .molecule import Kind
from .parameters import Types, orientations


@dataclass(frozen=True, slots=True)
class BondGroup:
    """Compare virtual bonds with both types in `members`; add a mismatch penalty.

    Groups with the same name are equivalent alternatives (e.g. two flavors of
    conjugated double bond). Their penalty must agree.
    """

    name: str
    members: frozenset[str]
    penalty: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", frozenset(self.members))
        if not self.name or not self.members or not isfinite(self.penalty) or self.penalty < 0:
            raise AutoCGENFFError("Invalid bond-group name, members or penalty")


@dataclass(frozen=True, slots=True)
class PenaltyModel:
    """Directional target-to-source matrices; missing off-diagonals are forbidden."""

    bonded: Mapping[tuple[str, str], float] = field(default_factory=dict)
    nonbonded: Mapping[tuple[str, str], float] = field(default_factory=dict)
    bond_groups: tuple[BondGroup, ...] = ()
    max_penalty: float | None = None

    def __post_init__(self) -> None:
        for name, mapping in (("bonded", self.bonded), ("nonbonded", self.nonbonded)):
            for (target, source), value in mapping.items():
                if target == "X" or source == "X" or not isfinite(value) or value < 0:
                    raise AutoCGENFFError(f"Invalid {name} substitution {target}->{source}")
                if target == source and value != 0:
                    raise AutoCGENFFError("Penalty matrix diagonals must be zero")
            object.__setattr__(self, name, MappingProxyType(dict(mapping)))
        object.__setattr__(self, "bond_groups", tuple(self.bond_groups))
        penalties: dict[str, float] = {}
        for group in self.bond_groups:
            if penalties.setdefault(group.name, group.penalty) != group.penalty:
                raise AutoCGENFFError(
                    f"Equivalent bond groups {group.name!r} need identical penalties"
                )
        if self.max_penalty is not None and (
            not isfinite(self.max_penalty) or self.max_penalty < 0
        ):
            raise AutoCGENFFError("Maximum penalty must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class Analogy:
    """Lowest-scoring candidate key and the orientation used to score it."""

    types: Types
    penalty: float
    reversed: bool


_INNER = {
    Kind.BOND: (0, 1),
    Kind.ANGLE: (1,),
    Kind.PROPER: (1, 2),
    Kind.NONBONDED: (),
    Kind.NBFIX: (),
}


def best_analogy(
    kind: Kind,
    target: Types,
    candidates: Iterable[Types],
    model: PenaltyModel,
    *,
    for_charge: bool = False,
    improper_center: int | None = None,
    source_improper_centers: Mapping[Types, int] | None = None,
) -> Analogy | None:
    """Find the deterministic lowest finite TPS, or None if no candidate is allowed.

    The query and source improper centers must align. If a source-center map
    is supplied, skip sources without a verified position (0 or 3); otherwise
    the caller must supply only center-first candidates. Nonbonded
    substitution is supported as a low-level score only.
    """
    if kind == Kind.IMPROPER and improper_center not in {0, 3}:
        raise AutoCGENFFError(
            "Improper analogy requires an explicit central atom position (0 or 3)"
        )
    best = None
    best_key = None
    for source in candidates:
        if "X" in source or len(source) != len(target):
            continue
        if kind == Kind.IMPROPER:
            source_center = (
                0 if source_improper_centers is None else source_improper_centers.get(source)
            )
            if source_center not in {0, 3}:
                continue
            reverse = source_center != improper_center
            ordered = ((source[::-1] if reverse else source, reverse),)
        else:
            ordered = orientations(kind, source)
        for oriented, reverse in ordered:
            score = _penalty(kind, target, oriented, model, for_charge, improper_center)
            if not isfinite(score) or (model.max_penalty is not None and score > model.max_penalty):
                continue
            key = (score, min(source, source[::-1]), source, reverse)
            if best_key is None or key < best_key:
                best_key = key
                best = Analogy(source, score, reverse)
    return best


def _penalty(
    kind: Kind,
    target: Types,
    source: Types,
    model: PenaltyModel,
    for_charge: bool,
    improper_center: int | None,
) -> float:
    inner = (improper_center,) if kind == Kind.IMPROPER else _INNER[kind]
    penalties = []
    for index, pair in enumerate(zip(target, source, strict=True)):
        requested, candidate = pair
        if requested == candidate:
            continue
        bonded = not for_charge and (index in inner or kind == Kind.IMPROPER)
        matrix = model.bonded if bonded else model.nonbonded
        if pair not in matrix:
            return inf
        penalties.append((10 if index in inner else 1) * matrix[pair])
    edges = []
    if kind in {Kind.BOND, Kind.ANGLE, Kind.PROPER}:
        for index in range(len(target) - 1):
            edges.append((index, index + 1, 1 if kind == Kind.PROPER and index != 1 else 10))
    elif kind == Kind.IMPROPER and improper_center is not None:
        edges.extend((improper_center, index, 1) for index in range(4) if index != improper_center)
    for left, right, weight in edges:
        first = _bond_memberships(target[left], target[right], model.bond_groups)
        second = _bond_memberships(source[left], source[right], model.bond_groups)
        penalties.extend(
            weight * (first | second)[key] for key in sorted(first.keys() ^ second.keys())
        )
    return fsum(penalties)


def _bond_memberships(left: str, right: str, groups: tuple[BondGroup, ...]) -> dict[str, float]:
    return {
        group.name: group.penalty
        for group in groups
        if left in group.members and right in group.members
    }
