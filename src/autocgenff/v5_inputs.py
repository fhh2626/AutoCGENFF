"""Optional native local inputs for v5 assignment. No data is downloaded or embedded."""

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from .analogy import (
    TYPE_RELATIONS,
    BondGroup,
    FallbackModel,
    PenaltyModel,
    _bond_memberships,
    type_relation,
)
from .charges import ChargeModel, reverse_increments
from .errors import AutoCGENFFError
from .molecule import Kind
from .parameters import _SECTIONS, _SHAPES, Parameters, Types
from .topology import Topology, improper_source_centers

_FIELD = re.compile(r"!\s*<%\s*(.*?)\s*%>\s*!")
_INCREMENT_KINDS = (Kind.BOND, Kind.ANGLE, Kind.PROPER)
# sp carbon types. A central torsion atom of these types lies on a 180 degree angle.
# NG1T1 is excluded: it also types the bent, carbon-bound N of an organic azide.
LINEAR_CENTERS = frozenset({"CG1N1", "CG1T1", "CG1T2"})


def read_native_increments(path: Path) -> ChargeModel:
    """Read oriented v5 !<% ... %>! fields, independent of the mechanical PRM.

    Missing fields are unresolved. Later Fourier components never introduce
    additional transfers. N/A is an observed noncontribution, distinct from zero.
    If at least one proper row has an sp carbon in a central position and every
    such row is annotated N/A, unlisted paths of this kind transfer nothing too.
    A numeric or unannotated row of this kind disables this rule.
    Raises AutoCGENFFError for malformed or conflicting annotations.
    """
    entries = {}
    rows = {}
    seen = set()
    shape = None
    unannotated_linear = False
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.partition("!")[0].upper().split()
            if not fields or fields[0].startswith("*"):
                continue
            head = fields[0]
            if head == "END":
                break
            if head in _SECTIONS or head in {"ATOMS", "CMAP", "HBOND", "THOLE"}:
                kind = _SECTIONS.get(head)
                shape = _SHAPES[kind] if kind in _INCREMENT_KINDS else None
                continue
            if shape is None:
                continue
            order, widths = shape
            types = tuple(fields[:order])
            key = min(types, types[::-1])
            duplicate = key in seen
            seen.add(key)
            m = _FIELD.search(line)
            if m is None:
                if "<%" in line or "%>" in line:
                    raise AutoCGENFFError(f"{path}:{number}: malformed increment delimiter")
                if order == 4 and LINEAR_CENTERS.intersection(types[1:3]):
                    unannotated_linear = True
                continue
            try:
                columns = fields[order:]
                if len(columns) not in widths or not all(
                    isfinite(float(value.replace("D", "E"))) for value in columns
                ):
                    raise ValueError("invalid parameter columns preceding increment annotation")
                body = m.group(1).strip().upper()
                values = (
                    None
                    if body == "N/A"
                    else tuple(float(x.replace("D", "E")) for x in body.split())
                )
                if (
                    "X" in key
                    or (
                        values is not None
                        and (len(values) != order - 1 or not all(map(isfinite, values)))
                    )
                ):
                    raise ValueError("invalid types or increment count/value")
                if duplicate and order == 4 and values is None:
                    continue  # Native file marks subsequent Fourier rows N/A.
                normal = values if key == types or values is None else reverse_increments(values)
                if duplicate and (key not in entries or entries[key] != normal):
                    raise ValueError("conflicting repeated increment annotation")
                if key == key[::-1] and normal is not None and normal != reverse_increments(normal):
                    raise ValueError("asymmetric palindromic increment")
                entries[key] = normal
                rows.setdefault(key, (types, number))
            except ValueError as exc:
                raise AutoCGENFFError(f"{path}:{number}: {types}: {exc}") from exc
    if not entries:
        raise AutoCGENFFError(f"No native v5 increment annotations in {path}")
    linear = [
        values for key, values in entries.items()
        if len(key) == 4 and LINEAR_CENTERS.intersection(key[1:3])
    ]
    native_na = linear and all(v is None for v in linear) and not unannotated_linear
    centers = LINEAR_CENTERS if native_na else frozenset()
    return ChargeModel(
        entries, source_rows=rows, infer_identical_bonds=False, linear_centers=centers
    )


# Conservative local-v1 analogy profile. Limit inference to substitutions within
# ordinary acyclic saturated carbons or their H; no bond-group changes, other
# elements, charged-adjacent, ring, aromatic, or cross-family guesses.
_FAMILIES = (frozenset("CG301 CG311 CG321 CG331".split()), frozenset("HGA1 HGA2 HGA3".split()))
_ANNOTATION = re.compile(r"\bfrom\s+([A-Z0-9 ]+?)\s*,\s*penalty\s*=\s*(\d+(?:\.\d+)?)\b", re.I)
_BONDED_KINDS = (Kind.BOND, Kind.ANGLE, Kind.PROPER, Kind.IMPROPER)
_ALTNUM_TWINS = (
    ("CG2DC1", "CG2DC2"),
    ("CG2D1O", "CG2D2O"),
    ("CG25C1", "CG25C2"),
    ("CG251O", "CG252O"),
)
_FALLBACK_SCALE = 20.0
_ELEMENT_MASSES = (
    (1.008, "H"),
    (10.811, "B"),
    (12.011, "C"),
    (14.007, "N"),
    (15.999, "O"),
    (18.998, "F"),
    (22.990, "Na"),
    (26.982, "Al"),
    (30.974, "P"),
    (32.060, "S"),
    (35.453, "Cl"),
    (40.080, "Ca"),
    (78.960, "Se"),
    (79.904, "Br"),
    (126.904, "I"),
)

# Published CGenFF Part II Table 3 bond groups (J. Chem. Inf. Model. 2012, 52, 3155).
_PART2_BOND_GROUPS = (
    BondGroup(
        "CONJUGATED_DOUBLE",
        frozenset("CG2DC1 CG2D1O CG25C1 CG251O CG2DC3 CG2D1 CG2D2 NG2D1 NG2P1".split()),
        40.0,
    ),
    BondGroup(
        "CONJUGATED_DOUBLE",
        frozenset("CG2DC2 CG2D2O CG25C2 CG252O CG2DC3 CG2D1 CG2D2 NG2D1 NG2P1".split()),
        40.0,
    ),
    BondGroup("PLANAR_7", frozenset("CG2R71 CG2RC7".split()), 40.0),
    BondGroup(
        "PLANAR_6",
        frozenset(
            "CG2R61 CG2R62 CG2R63 CG2R64 CG2R66 NG2R60 NG2R61 NG2R62 CG2RC0 CG2R67".split()
        ),
        20.0,
    ),
    BondGroup(
        "PLANAR_5",
        frozenset(
            "CG2R51 CG2R52 CG2R53 NG2R50 NG2R51 NG2R52 NG2R53 OG2R50 SG2R50".split()
        ),
        20.0,
    ),
    BondGroup(
        "RING_5",
        frozenset(
            "CG3C50 CG3C51 CG3C52 CG3C53 CG3C54 NG3C51 OG3C51 CG3RC1 CG2R51 CG2R52 "
            "CG2R53 NG2R50 NG2R51 NG2R52 NG2R53 OG2R50 SG2R50 CG25C1 CG25C2 CG251O "
            "CG252O CG2RC0 NG2RC0 CG2RC7".split()
        ),
        20.0,
    ),
    BondGroup("RING_4", frozenset("CG3C41 CG3RC1".split()), 60.0),
    BondGroup("RING_3", frozenset("CG3C31 CG3RC1".split()), 80.0),
    BondGroup("BIPHENYL", frozenset(("CG2R67",)), 47.0),
    BondGroup("TRIPLE", frozenset("CG1T1 CG1N1 NG1T1".split()), 27.0),
)

# Published CGenFF Part II Table 1 NG3 bonded hierarchy (two printed self-alt typos corrected).
_NG3P_PRI = {"NG3P2": 0.0, "NG3P3": 1.0, "NG3P1": 3.0, "NG3P0": 4.0}
_NG3P_ALTS = {
    ("NG3P2", "NG3P1"): 1.0, ("NG3P2", "NG3P3"): 3.0, ("NG3P2", "NG3P0"): 4.0,
    ("NG3P3", "NG3P2"): 1.0, ("NG3P3", "NG3P1"): 2.0, ("NG3P3", "NG3P0"): 4.0,
    ("NG3P1", "NG3P2"): 1.0, ("NG3P1", "NG3P0"): 3.0, ("NG3P1", "NG3P3"): 4.0,
    ("NG3P0", "NG3P1"): 1.0, ("NG3P0", "NG3P2"): 2.0, ("NG3P0", "NG3P3"): 4.0,
}
_NG3N_PRI = {
    "NG321": 0.0, "NG311": 0.5, "NG301": 1.0, "NG3N1": 1.5, "NG3C51": 2.5, "NG331": 4.0,
}
_NG3N_ALTS = {
    ("NG321", "NG311"): 1.0, ("NG321", "NG301"): 1.5, ("NG321", "NG3N1"): 2.5,
    ("NG321", "NG3C51"): 3.0, ("NG321", "NG331"): 4.0,
    ("NG311", "NG301"): 0.5, ("NG311", "NG321"): 1.0, ("NG311", "NG3N1"): 1.5,
    ("NG311", "NG3C51"): 2.0, ("NG311", "NG331"): 5.0,
    ("NG301", "NG311"): 0.5, ("NG301", "NG321"): 1.5, ("NG301", "NG3N1"): 2.0,
    ("NG301", "NG3C51"): 2.5, ("NG301", "NG331"): 5.5,
    ("NG3N1", "NG311"): 1.5, ("NG3N1", "NG301"): 2.0, ("NG3N1", "NG321"): 2.5,
    ("NG3N1", "NG3C51"): 3.5, ("NG3N1", "NG331"): 6.5,
    ("NG3C51", "NG311"): 2.0, ("NG3C51", "NG301"): 2.5, ("NG3C51", "NG321"): 3.0,
    ("NG3C51", "NG3N1"): 4.0, ("NG3C51", "NG331"): 7.0,
    ("NG331", "NG321"): 4.0, ("NG331", "NG311"): 5.0, ("NG331", "NG301"): 5.5,
    ("NG331", "NG3N1"): 6.5, ("NG331", "NG3C51"): 7.0,
}


def _ng3_hierarchy() -> dict[tuple[str, str], float]:
    table = {**_NG3P_ALTS, **_NG3N_ALTS}
    for first in _NG3P_PRI:
        for second, pri in _NG3N_PRI.items():
            table[(first, second)] = 10.0 + pri
    for first in _NG3N_PRI:
        for second, pri in _NG3P_PRI.items():
            table[(first, second)] = 10.0 + pri
    return table


def _position_roles(kind: Kind, center: int | None) -> tuple[tuple[str, float], ...]:
    if kind == Kind.BOND:
        return (("bonded", 10.0), ("bonded", 10.0))
    if kind == Kind.ANGLE:
        return (("nonbonded", 1.0), ("bonded", 10.0), ("nonbonded", 1.0))
    if kind == Kind.PROPER:
        return (("nonbonded", 1.0), ("bonded", 10.0), ("bonded", 10.0), ("nonbonded", 1.0))
    return tuple(("bonded", 10.0 if i == center else 1.0) for i in range(4))


def _bond_group_penalty(kind: Kind, target: Types, source: Types) -> float:
    if kind not in {Kind.BOND, Kind.ANGLE, Kind.PROPER}:
        return 0.0
    total = 0.0
    for pos in range(len(target) - 1):
        weight = 1.0 if kind == Kind.PROPER and pos != 1 else 10.0
        first = _bond_memberships(target[pos], target[pos + 1], _PART2_BOND_GROUPS)
        second = _bond_memberships(source[pos], source[pos + 1], _PART2_BOND_GROUPS)
        if first != second:
            total += sum(
                weight * (first | second)[key] for key in first.keys() ^ second.keys()
            )
    return total


def _elements_from_topology(topology: Topology | None) -> dict[str, str]:
    elements: dict[str, str] = {}
    if topology is None:
        return elements
    for atom_type, mass in topology.masses.items():
        if atom_type in topology.elements:
            elements[atom_type] = topology.elements[atom_type]
        elif mass > 0.5:
            best_mass, symbol = min(_ELEMENT_MASSES, key=lambda row: abs(row[0] - mass))
            if abs(best_mass - mass) < 0.05:
                elements[atom_type] = symbol
    return elements


def _mode_cost(values: list[float]) -> float:
    counts = Counter(values)
    best = max(counts.values())
    return min(value for value, count in counts.items() if count == best)


def _best_source_orientation(target: Types, source: Types) -> tuple[list[int], Types]:
    best_diffs: list[int] | None = None
    best_cand = source
    for candidate in (source, source[::-1]):
        diffs = [i for i, (a, b) in enumerate(zip(target, candidate, strict=True)) if a != b]
        if best_diffs is None or len(diffs) < len(best_diffs):
            best_diffs = diffs
            best_cand = candidate
    return best_diffs or [], best_cand


@dataclass(frozen=True, slots=True)
class SubstitutionEvidence:
    """One selected directional cost and a supporting line of the caller's PRM."""

    matrix: str
    target: str
    source: str
    cost: float
    line: int


def observed_alkyl_model(
    parameters: Parameters,
) -> tuple[PenaltyModel, tuple[SubstitutionEvidence, ...]]:
    """Infer only directionally observed, single-position substitution costs.

    A local PRM annotation needs a real oriented source row and exactly one
    unambiguous mismatch. Use the mode (lower cost breaks ties)
    across independent parameter groups. All other off-diagonals are forbidden.
    Inspired by LigandFF's provisional single-substitution observations (MIT).
    """
    selected = {
        group[0].line: (kind, group[0].types)
        for kind in _INCREMENT_KINDS
        for group in parameters.iter_groups(kind)
    }
    observed = defaultdict(lambda: defaultdict(list))
    with parameters.source.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if line_no not in selected:
                continue
            kind, types = selected[line_no]
            match = _ANNOTATION.search(line.partition("!")[2])
            if match is None:
                continue
            source = tuple(match[1].upper().split())
            if len(source) != len(types) or "X" in source or "X" in types:
                continue
            orientations = set()
            for candidate in (source, source[::-1]):
                changes = [
                    i for i, (a, b) in enumerate(zip(types, candidate, strict=True)) if a != b
                ]
                if len(changes) == 1:
                    i = changes[0]
                    orientations.add((i, types[i], candidate[i]))
            if len(orientations) != 1:
                continue
            i, a, b = orientations.pop()
            if not any({a, b} <= family for family in _FAMILIES):
                continue
            source_match = parameters.match(kind, source)
            if source_match is None or source_match.wildcard:
                continue
            inner = kind == Kind.BOND or 0 < i < len(types) - 1
            cost = float(match[2]) / (10 if inner else 1)
            observed[("bonded" if inner else "nonbonded", a, b)][cost].append(line_no)
    matrices = {"bonded": {}, "nonbonded": {}}
    evidence = []
    for (matrix, a, b), votes in sorted(observed.items()):
        cost = min(votes, key=lambda value: (-len(votes[value]), value))
        matrices[matrix][(a, b)] = cost
        evidence.append(SubstitutionEvidence(matrix, a, b, cost, votes[cost][0]))
    return PenaltyModel(**matrices), tuple(evidence)


def observed_v5_penalty_model(
    parameters: Parameters, topology: Topology | None = None
) -> tuple[PenaltyModel, tuple[SubstitutionEvidence, ...]]:
    """Build a CGenFF Part II directional penalty model from local PRM annotations.

    Combines published Part II Table 1 (NG3 hierarchy) and Table 3 bond groups
    with single-substitution observations, iterative multi-substitution solving,
    zero-cost altnum twins, and scaled relation fallbacks. Inspired by
    LigandFF's Part II penalty fitting (MIT).
    """
    selected = {}
    for kind in _BONDED_KINDS:
        for group in parameters.iter_groups(kind):
            for row in group:
                selected[row.line] = kind, row.types, group[0].line
    centers = improper_source_centers(topology) if topology is not None else {}
    annotations: list[tuple[Kind, Types, Types, float, int]] = []
    seen_groups: set[int] = set()
    with parameters.source.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if line_no not in selected:
                continue
            kind, types, group_line = selected[line_no]
            match = _ANNOTATION.search(line.partition("!")[2])
            if match is None or group_line in seen_groups:
                continue
            source = tuple(match[1].upper().split())
            if len(source) != len(types) or "X" in source or "X" in types:
                continue
            source_match = parameters.match(kind, source)
            tps = float(match[2])
            if source_match is None or source_match.wildcard or not isfinite(tps):
                continue
            if kind == Kind.IMPROPER and not (types in centers and source in centers):
                continue
            # A multi-row family supplies one observation, from its first usable note.
            seen_groups.add(group_line)
            annotations.append((kind, types, source, tps, line_no))

    if not annotations:
        return PenaltyModel(), ()

    observed: dict[tuple[str, str, str], dict[float, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    equations: list[tuple[dict[tuple[str, str, str], float], float, int]] = []
    for kind, types, source, tps, line_no in annotations:
        center = centers[types] if kind == Kind.IMPROPER else None
        if center is None:
            diffs, cand = _best_source_orientation(types, source)
        else:
            cand = source if center == centers[source] else source[::-1]
            diffs = [i for i, (a, b) in enumerate(zip(types, cand, strict=True)) if a != b]
        if not diffs:
            continue
        roles = _position_roles(kind, center)
        group_pen = _bond_group_penalty(kind, types, cand)
        if len(diffs) == 1 and group_pen == 0.0:
            pos = diffs[0]
            matrix, weight = roles[pos]
            cost = round(tps / weight, 4)
            observed[(matrix, types[pos], cand[pos])][cost].append(line_no)
        elif group_pen <= tps:
            terms: dict[tuple[str, str, str], float] = defaultdict(float)
            for pos in diffs:
                matrix, weight = roles[pos]
                terms[(matrix, types[pos], cand[pos])] += weight
            equations.append((terms, tps - group_pen, line_no))

    hierarchy = _ng3_hierarchy()
    matrices: dict[str, dict[tuple[str, str], float]] = {"bonded": dict(hierarchy), "nonbonded": {}}
    evidence: list[SubstitutionEvidence] = []
    # Keep single-substitution observations for fallback medians before adding twins/solved entries.
    single_observed: dict[str, dict[tuple[str, str], float]] = {"bonded": {}, "nonbonded": {}}
    for (matrix, a, b), votes in sorted(observed.items()):
        cost = min(votes, key=lambda value: (-len(votes[value]), value))
        single_observed[matrix][(a, b)] = cost
        if matrix == "bonded" and (a, b) in hierarchy:
            continue
        matrices[matrix][(a, b)] = cost
        evidence.append(SubstitutionEvidence(matrix, a, b, cost, votes[cost][0]))

    for a, b in _ALTNUM_TWINS:
        for matrix in ("bonded", "nonbonded"):
            matrices[matrix][(a, b)] = 0.0
            matrices[matrix][(b, a)] = 0.0

    # Iteratively solve multi-substitution annotations when all but one term is known.
    while True:
        solved_votes: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
        for terms, rhs, line_no in equations:
            unknown = [
                (entry, weight) for entry, weight in terms.items()
                if (entry[1], entry[2]) not in matrices[entry[0]]
            ]
            if len(unknown) != 1:
                continue
            (entry, weight) = unknown[0]
            known_sum = sum(
                w * matrices[e[0]][(e[1], e[2])] for e, w in terms.items() if e != entry
            )
            value = round((rhs - known_sum) / weight, 4)
            if value >= 0.0:
                solved_votes[entry].append((value, line_no))
        if not solved_votes:
            break
        for (matrix, a, b), pairs in sorted(solved_votes.items()):
            cost = _mode_cost([value for value, _ in pairs])
            matrices[matrix][(a, b)] = cost
            first_line = min(line_no for value, line_no in pairs if value == cost)
            evidence.append(SubstitutionEvidence(matrix, a, b, cost, first_line))

    elements = _elements_from_topology(topology)
    fallback_tables: dict[str, dict[tuple[str, ...], float]] = {}
    for matrix in ("bonded", "nonbonded"):
        by_rel: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for (a, b), value in single_observed[matrix].items():
            by_rel[type_relation(a, b, elements)].append(value)
        medians = {rel: statistics.median(vals) for rel, vals in by_rel.items()}
        medians.setdefault(("diff",), max(single_observed[matrix].values(), default=10.0))
        table: dict[tuple[str, ...], float] = {}
        farther = medians[("diff",)]
        for rel in reversed(TYPE_RELATIONS):
            farther = table[rel] = medians.get(rel, farther)
        for level in ("1", "0"):
            prev = str(int(level) + 1)
            table[("fam", level)] = max(table[("fam", level)], table[("fam", prev)])
        fallback_tables[matrix] = {rel: value * _FALLBACK_SCALE for rel, value in table.items()}

    fallback = (
        FallbackModel(fallback_tables["bonded"], fallback_tables["nonbonded"], elements)
        if single_observed["bonded"] or single_observed["nonbonded"]
        else None
    )
    model = PenaltyModel(
        matrices["bonded"],
        matrices["nonbonded"],
        _PART2_BOND_GROUPS,
        fallback=fallback,
        improper_bond_groups=False,
    )
    return model, tuple(evidence)
