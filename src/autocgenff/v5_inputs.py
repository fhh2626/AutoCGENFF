"""Optional native local inputs for v5 assignment. No data is downloaded or embedded."""

import re
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from .analogy import PenaltyModel
from .charges import ChargeModel, reverse_increments
from .errors import AutoCGENFFError
from .molecule import Kind
from .parameters import _SECTIONS, _SHAPES, Parameters

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
