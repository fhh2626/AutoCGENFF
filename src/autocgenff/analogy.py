"""Explicit opt-in directional penalties for transferable parameter candidates."""

import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from math import fsum, inf, isfinite
from types import MappingProxyType

from .errors import AutoCGENFFError
from .molecule import Kind
from .parameters import Types, orientations

_TYPE_FAMILY = re.compile(r"^([A-Z]+?G.?)(.*)$")
TYPE_RELATIONS: tuple[tuple[str, ...], ...] = (
    ("fam", "2"),
    ("fam", "1"),
    ("fam", "0"),
    ("elem", "H"),
    ("elem", "other"),
    ("diff",),
)


def type_relation(target: str, source: str, elements: Mapping[str, str]) -> tuple[str, ...]:
    """Classify similarity between two CHARMM atom types for fallback costing."""
    first, second = elements.get(target), elements.get(source)
    if first is None or second is None or first != second:
        return ("diff",)
    left, right = _TYPE_FAMILY.match(target), _TYPE_FAMILY.match(source)
    if left is None or right is None or left.group(1) != right.group(1):
        return ("elem", "H" if first == "H" else "other")
    tail_a, tail_b = left.group(2), right.group(2)
    shared = 0
    while shared < min(len(tail_a), len(tail_b), 2) and tail_a[shared] == tail_b[shared]:
        shared += 1
    return ("fam", str(shared))


@dataclass(frozen=True, slots=True)
class FallbackModel:
    """Optional relation-based fallback penalties for unobserved substitutions."""

    bonded: Mapping[tuple[str, ...], float]
    nonbonded: Mapping[tuple[str, ...], float]
    elements: Mapping[str, str]

    def __post_init__(self) -> None:
        for name, table in (("bonded", self.bonded), ("nonbonded", self.nonbonded)):
            for key, value in table.items():
                if not isfinite(value) or value < 0:
                    raise AutoCGENFFError(f"Invalid fallback {name} penalty for {key}")
            object.__setattr__(self, name, MappingProxyType(dict(table)))
        object.__setattr__(self, "elements", MappingProxyType(dict(self.elements)))

    def cost(self, matrix: str, target: str, source: str) -> float:
        table = self.bonded if matrix == "bonded" else self.nonbonded
        return table.get(type_relation(target, source, self.elements), inf)


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
    """Directional target-to-source matrices; missing off-diagonals are forbidden unless fallback is set."""

    bonded: Mapping[tuple[str, str], float] = field(default_factory=dict)
    nonbonded: Mapping[tuple[str, str], float] = field(default_factory=dict)
    bond_groups: tuple[BondGroup, ...] = ()
    max_penalty: float | None = None
    fallback: FallbackModel | None = None
    improper_bond_groups: bool = True

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

    def substitution_cost(self, matrix: str, target: str, source: str) -> float:
        """Return directional substitution penalty, or inf when forbidden."""
        if target == source:
            return 0.0
        table = self.bonded if matrix == "bonded" else self.nonbonded
        value = table.get((target, source))
        if value is not None:
            return value
        if self.fallback is not None:
            return self.fallback.cost(matrix, target, source)
        return inf


@dataclass(frozen=True, slots=True)
class Analogy:
    """Lowest-scoring candidate key and the orientation used to score it."""

    types: Types
    penalty: float
    reversed: bool


_ARITY = {
    Kind.BOND: 2,
    Kind.ANGLE: 3,
    Kind.PROPER: 4,
    Kind.IMPROPER: 4,
    Kind.NONBONDED: 1,
    Kind.NBFIX: 2,
}
_INNER = {
    Kind.BOND: (0, 1),
    Kind.ANGLE: (1,),
    Kind.PROPER: (1, 2),
    Kind.NONBONDED: (),
    Kind.NBFIX: (),
}
_INNER_BONDS = {
    Kind.BOND: ((0, 1, 10),),
    Kind.ANGLE: (),
    Kind.PROPER: ((1, 2, 10),),
    Kind.NONBONDED: (),
    Kind.NBFIX: (),
}


class CandidateIndex:
    """Exact inner-atom bucket index for pruning analogy searches."""

    __slots__ = ("kind", "improper_center", "buckets")

    def __init__(
        self,
        kind: Kind,
        candidates: Iterable[Types],
        *,
        improper_center: int | None = None,
        source_improper_centers: Mapping[Types, int] | None = None,
    ) -> None:
        self.kind = kind
        self.improper_center = improper_center
        expected_len = _ARITY[kind]
        inner = (improper_center,) if kind == Kind.IMPROPER else _INNER[kind]
        grouped: dict[Types, list[tuple[Types, Types, bool, Types]]] = defaultdict(list)
        for source in candidates:
            if "X" in source or len(source) != expected_len:
                continue
            if kind == Kind.IMPROPER:
                if improper_center not in {0, 3}:
                    continue
                source_center = (
                    0 if source_improper_centers is None else source_improper_centers.get(source)
                )
                if source_center not in {0, 3}:
                    continue
                reverse = source_center != improper_center
                ordered = ((source[::-1] if reverse else source, reverse),)
            else:
                ordered = orientations(kind, source)
            canonical = min(source, source[::-1])
            for oriented, reverse in ordered:
                inner_key = tuple(oriented[index] for index in inner)
                grouped[inner_key].append((oriented, source, reverse, canonical))
        self.buckets: dict[Types, tuple[tuple[Types, Types, bool, Types], ...]] = {
            key: tuple(rows) for key, rows in grouped.items()
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
    candidate_index: CandidateIndex | None = None,
    exclude: Collection[Types] = (),
) -> Analogy | None:
    """Find the deterministic lowest finite TPS, or None if no candidate is allowed.

    The query and source improper centers must align. If a source-center map
    is supplied, skip sources without a verified position (0 or 3); otherwise
    the caller must supply only center-first candidates. Nonbonded
    substitution is supported as a low-level score only.
    """
    if len(target) != _ARITY[kind] or "X" in target:
        raise AutoCGENFFError(f"Invalid type tuple for {kind}: {target}")
    if kind == Kind.IMPROPER and improper_center not in {0, 3}:
        raise AutoCGENFFError(
            "Improper analogy requires an explicit central atom position (0 or 3)"
        )
    if candidate_index is not None:
        if candidate_index.kind != kind:
            raise AutoCGENFFError(
                f"CandidateIndex kind {candidate_index.kind} does not match query kind {kind}"
            )
        if kind == Kind.IMPROPER and candidate_index.improper_center != improper_center:
            raise AutoCGENFFError(
                "CandidateIndex improper_center does not match query improper_center"
            )
        return _best_indexed(
            kind, target, candidate_index, model, for_charge, improper_center, exclude
        )
    best = None
    best_key = None
    for source in candidates:
        if "X" in source or len(source) != len(target):
            continue
        canonical = min(source, source[::-1])
        if exclude and canonical in exclude:
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
            key = (score, canonical, source, reverse)
            if best_key is None or key < best_key:
                best_key = key
                best = Analogy(source, score, reverse)
    return best


def _best_indexed(
    kind: Kind,
    target: Types,
    index: CandidateIndex,
    model: PenaltyModel,
    for_charge: bool,
    improper_center: int | None,
    exclude: Collection[Types] = (),
) -> Analogy | None:
    inner = (improper_center,) if kind == Kind.IMPROPER else _INNER[kind]
    inner_matrix = "nonbonded" if for_charge else "bonded"
    inner_bonds = () if kind == Kind.IMPROPER else _INNER_BONDS[kind]
    target_len = len(target)
    target_groups = {}
    if model.bond_groups:
        if kind in {Kind.BOND, Kind.ANGLE, Kind.PROPER}:
            for pos in range(target_len - 1):
                target_groups[(pos, pos + 1)] = _bond_memberships(
                    target[pos], target[pos + 1], model.bond_groups
                )
        elif kind == Kind.IMPROPER and model.improper_bond_groups and improper_center is not None:
            for pos in range(4):
                if pos != improper_center:
                    target_groups[(improper_center, pos)] = _bond_memberships(
                        target[improper_center], target[pos], model.bond_groups
                    )
    bounds: list[tuple[float, Types]] = []
    max_pen = model.max_penalty
    for inner_key in index.buckets:
        if len(inner_key) != len(inner):
            continue
        lb_parts = []
        valid = True
        for pos, cand_type in zip(inner, inner_key, strict=True):
            cost = model.substitution_cost(inner_matrix, target[pos], cand_type)
            if not isfinite(cost):
                valid = False
                break
            lb_parts.append(10.0 * cost)
        if not valid:
            continue
        if model.bond_groups and inner_bonds:
            for left, right, weight in inner_bonds:
                first = target_groups[(left, right)]
                second = _bond_memberships(
                    inner_key[inner.index(left)],
                    inner_key[inner.index(right)],
                    model.bond_groups,
                )
                if first != second:
                    lb_parts.extend(
                        weight * (first | second)[g] for g in sorted(first.keys() ^ second.keys())
                    )
        lb = fsum(lb_parts)
        if max_pen is not None and lb > max_pen:
            continue
        bounds.append((lb, inner_key))
    bounds.sort(key=lambda item: item[0])
    best: Analogy | None = None
    best_key: tuple[float, Types, Types, bool] | None = None
    for lb, inner_key in bounds:
        if best_key is not None and lb > best_key[0] + 1e-12:
            break
        for oriented, source, reverse, canonical in index.buckets[inner_key]:
            if len(oriented) != target_len or (exclude and canonical in exclude):
                continue
            score = _penalty(
                kind, target, oriented, model, for_charge, improper_center, target_groups
            )
            if not isfinite(score) or (max_pen is not None and score > max_pen):
                continue
            key = (score, canonical, source, reverse)
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
    target_groups: Mapping[tuple[int, int], dict[str, float]] | None = None,
) -> float:
    inner = (improper_center,) if kind == Kind.IMPROPER else _INNER[kind]
    penalties = []
    for index, pair in enumerate(zip(target, source, strict=True)):
        requested, candidate = pair
        if requested == candidate:
            continue
        matrix = "bonded" if not for_charge and (index in inner or kind == Kind.IMPROPER) else "nonbonded"
        cost = model.substitution_cost(matrix, requested, candidate)
        if not isfinite(cost):
            return inf
        penalties.append((10 if index in inner else 1) * cost)
    if not model.bond_groups:
        return fsum(penalties)
    edges = []
    if kind in {Kind.BOND, Kind.ANGLE, Kind.PROPER}:
        for index in range(len(target) - 1):
            edges.append((index, index + 1, 1 if kind == Kind.PROPER and index != 1 else 10))
    elif kind == Kind.IMPROPER and model.improper_bond_groups and improper_center is not None:
        edges.extend((improper_center, index, 1) for index in range(4) if index != improper_center)
    for left, right, weight in edges:
        first = (
            target_groups[(left, right)]
            if target_groups is not None
            else _bond_memberships(target[left], target[right], model.bond_groups)
        )
        second = _bond_memberships(source[left], source[right], model.bond_groups)
        if first != second:
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
