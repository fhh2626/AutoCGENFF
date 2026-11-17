"""Read and match an additive CHARMM parameter file, retaining source lines."""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite
from pathlib import Path

from .errors import AutoCGENFFError
from .molecule import Kind

Types = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    """One CHARMM parameter row, with values in the original column order."""

    types: Types
    values: tuple[float, ...]
    line: int


@dataclass(frozen=True, slots=True)
class ParameterMatch:
    """The selected records and their orientation relative to the query."""

    records: tuple[ParameterRecord, ...]
    record_reversed: tuple[bool, ...]
    wildcard: bool = False
    reversed: bool | None = field(init=False)

    def __post_init__(self) -> None:
        # Keep the group flag for uniform matches. Mixed groups have no single orientation.
        directions = set(self.record_reversed)
        object.__setattr__(self, "reversed", directions.pop() if len(directions) == 1 else None)


_SECTIONS = {
    "BOND": Kind.BOND,
    "BONDS": Kind.BOND,
    "ANGL": Kind.ANGLE,
    "ANGLE": Kind.ANGLE,
    "ANGLES": Kind.ANGLE,
    "THETA": Kind.ANGLE,
    "THETAS": Kind.ANGLE,
    "DIHE": Kind.PROPER,
    "DIHEDRAL": Kind.PROPER,
    "DIHEDRALS": Kind.PROPER,
    "PHI": Kind.PROPER,
    "IMPR": Kind.IMPROPER,
    "IMPROPER": Kind.IMPROPER,
    "IMPROPERS": Kind.IMPROPER,
    "IMPH": Kind.IMPROPER,
    "IMPHI": Kind.IMPROPER,
    "NONB": Kind.NONBONDED,
    "NBON": Kind.NONBONDED,
    "NBOND": Kind.NONBONDED,
    "NBONDED": Kind.NONBONDED,
    "NONBONDED": Kind.NONBONDED,
    "NBFIX": Kind.NBFIX,
}
_SHAPES = {
    Kind.BOND: (2, (2,)),
    Kind.ANGLE: (3, (2, 4)),
    Kind.PROPER: (4, (3,)),
    Kind.IMPROPER: (4, (3,)),
    Kind.NONBONDED: (1, (3, 6)),
    Kind.NBFIX: (2, (2, 3, 4)),
}
_NONBONDED_OPTIONS = frozenset(
    "cutnb ctofnb ctonnb eps e14fac wmin nbxmod atom group switch vswitch vfswitch shift vshift fshift bycb bycc vswitched".upper().split()
)


# c47b1 I/O manual: A-X-X-B, X-A-B-C, X-A-B-X, then X-X-A-B (either direction).
_IMPROPER_WILDCARDS = ((1, 2), (0,), (0, 3), (0, 1))


def _improper_wildcard_rank(types: Types) -> int:
    positions = tuple(i for i, atom_type in enumerate(types) if atom_type == "X")
    reverse = tuple(3 - i for i in reversed(positions))
    for rank, pattern in enumerate(_IMPROPER_WILDCARDS):
        if pattern in (positions, reverse):
            return rank
    return len(_IMPROPER_WILDCARDS)


# CHARMM type patterns are supported in both NONB and NBFIX.
_TYPE_METACHARACTERS = {"*": ".*", "%": ".", "#": "[0-9]*", "+": "[0-9]"}
_PRM_NUMBER_TOKEN = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[ED][+-]?[0-9]+)?|INF(?:INITY)?|NAN)"
)


def _has_type_pattern(pattern: str) -> bool:
    return any(character in _TYPE_METACHARACTERS for character in pattern)


@lru_cache(maxsize=512)
def _type_pattern_regex(pattern: str) -> re.Pattern[str]:
    return re.compile("".join(_TYPE_METACHARACTERS.get(char, re.escape(char)) for char in pattern))


def _matches_nbfix_type(pattern: str, atom_type: str) -> bool:
    # NBFIX also accepts the legacy, whole-field X wildcard.
    return (
        pattern == "X"
        or pattern == atom_type
        or (
            _has_type_pattern(pattern)
            and _type_pattern_regex(pattern).fullmatch(atom_type) is not None
        )
    )


def _validate_wildcards(kind: Kind, types: Types) -> None:
    if kind == Kind.NBFIX or "X" not in types:
        return
    if kind == Kind.PROPER:
        if types[0] != "X" or types[-1] != "X" or "X" in types[1:-1]:
            raise ValueError("proper wildcard layout must be X-A-B-X")
    elif kind == Kind.IMPROPER:
        if _improper_wildcard_rank(types) == len(_IMPROPER_WILDCARDS):
            raise ValueError("unsupported improper wildcard layout")
    else:
        raise ValueError("X wildcards supported only for dihedrals, impropers and NBFIX")


def charmm_lines(
    path: Path, *, asterisk_row: Callable[[list[str], list[str], bool], bool] | None = None
) -> Iterator[tuple[int, list[str]]]:
    """Iterate upper-case CHARMM tokens, joining continuation lines and skipping display control."""
    parts = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.split("!", 1)[0].strip()
            if not text:
                continue
            continued = text.endswith("-") and len(text) > 1 and text[-2].isspace()
            tokens = (text[:-1] if continued else text).upper().split()
            if text.startswith("*") and (
                asterisk_row is None or not asterisk_row(parts, tokens, continued)
            ):
                continue
            if not parts:
                first_line = line_no
            parts.extend(tokens)
            if not continued:
                if parts[0] not in {"PRIN", "PRINT"} or parts[1:] not in ([], ["ON"], ["OFF"]):
                    yield first_line, parts
                parts = []
    if parts:
        raise AutoCGENFFError(f"{path}:{first_line}: unterminated continuation")


def _key(kind: Kind, types: Types) -> Types:
    if kind == Kind.IMPROPER and "X" in types:
        return types  # Keep wildcard rows separate for file-order tie-breaking.
    return min(types, types[::-1])


def orientations(kind: Kind, types: Types) -> tuple[tuple[Types, bool], ...]:
    """Return permitted orientations for a type tuple (no arbitrary permutations)."""
    if len(types) == 1 or types == types[::-1]:
        return ((types, False),)
    return ((types, False), (types[::-1], True))


class Parameters:
    """Read-only lookup built by `read_parameters`; no substitutions are implicit."""

    def __init__(
        self,
        source: Path,
        entries: dict[Kind, dict[Types, tuple[ParameterRecord, ...]]],
        unsupported_sections: tuple[str, ...],
    ) -> None:
        self.source = source
        self.unsupported_sections = unsupported_sections
        self._entries = {
            kind: dict(sorted(groups.items(), key=lambda entry: entry[1][0].line))
            for kind, groups in entries.items()
        }
        self._nbfix = tuple(
            sorted((group[0] for group in self.iter_groups(Kind.NBFIX)), key=lambda row: row.line)
        )
        self._nonbonded_patterns = tuple(
            sorted(
                (
                    group[0]
                    for group in self.iter_groups(Kind.NONBONDED)
                    if _has_type_pattern(group[0].types[0])
                ),
                key=lambda row: row.line,
            )
        )
        self._wildcards = {
            kind: tuple(group for group in groups.values() if "X" in group[0].types)
            for kind, groups in self._entries.items()
        }
        if Kind.IMPROPER in self._wildcards:
            self._wildcards[Kind.IMPROPER] = tuple(
                sorted(
                    self._wildcards[Kind.IMPROPER],
                    key=lambda group: _improper_wildcard_rank(group[0].types),
                )
            )

    def iter_groups(self, kind: Kind) -> Iterator[tuple[ParameterRecord, ...]]:
        """Iterate parsed record groups in input order."""
        yield from self._entries.get(kind, {}).values()

    def match(self, kind: Kind, types: Types) -> ParameterMatch | None:
        """Use exact then wildcard matches, except NBFIX uses last applicable row."""
        if len(types) != _SHAPES[kind][0] or "X" in types:
            raise AutoCGENFFError(f"Invalid type tuple for {kind}: {types}")
        if kind == Kind.NBFIX:
            for row in reversed(self._nbfix):
                left, right = row.types
                for (first, second), reverse in orientations(kind, types):
                    if _matches_nbfix_type(left, first) and _matches_nbfix_type(right, second):
                        wildcard = "X" in row.types or any(map(_has_type_pattern, row.types))
                        return ParameterMatch((row,), (reverse,), wildcard=wildcard)
            return None
        entries = self._entries.get(kind, {})
        exact = entries.get(_key(kind, types))
        if exact is not None:
            return ParameterMatch(exact, tuple(row.types != types for row in exact))
        if kind == Kind.NONBONDED:
            for row in reversed(self._nonbonded_patterns):
                if _type_pattern_regex(row.types[0]).fullmatch(types[0]) is not None:
                    return ParameterMatch((row,), (False,), wildcard=True)
            return None
        for group in self._wildcards.get(kind, ()):
            for ordered, reverse in orientations(kind, types):
                if all(
                    expected in (actual, "X")
                    for expected, actual in zip(group[0].types, ordered, strict=True)
                ):
                    flags = tuple(reverse != (row.types != group[0].types) for row in group)
                    return ParameterMatch(group, flags, wildcard=True)
        return None

    def nbfix_for(self, types: Types) -> tuple[ParameterRecord, ...]:
        """Return applicable native pair override rows in source order, without expansion."""
        present = frozenset(types)
        return tuple(
            row
            for row in self._nbfix
            if all(
                any(_matches_nbfix_type(pattern, atom_type) for atom_type in present)
                for pattern in row.types
            )
        )


def read_parameters(path: Path) -> Parameters:
    """Parse a standalone additive CHARMM PRM; malformed supported rows raise."""
    kind = None
    entries: dict[Kind, dict[Types, list[ParameterRecord]]] = {}
    unsupported = set()

    def asterisk_row(parts: list[str], tokens: list[str], continued: bool) -> bool:
        if kind not in {Kind.NONBONDED, Kind.NBFIX}:
            return False
        arity, _ = _SHAPES[kind]
        if len(parts) >= arity:
            return False
        # A compact *L pattern belongs to the row even if the numeric columns
        # are malformed. A bare * is also a title marker, so require row context.
        first = tokens[0]
        if first != "*":
            return re.fullmatch(r"\*(?=[A-Z0-9%#+*]*[A-Z0-9%#+])[A-Z0-9%#+*]+", first) is not None
        if continued and len(parts) + len(tokens) <= arity:
            return True
        values = (parts + tokens)[arity:]
        # A bare-star title can contain a year. Require a numeric column run,
        # leaving count and finiteness validation to the parameter reader.
        return bool(values) and all(_PRM_NUMBER_TOKEN.fullmatch(value) for value in values)

    for line_no, fields in charmm_lines(path, asterisk_row=asterisk_row):
        head = fields[0]
        if head == "END":
            break
        if head in _SECTIONS:
            kind = _SECTIONS[head]
            entries.setdefault(kind, {})
            continue
        if head in {"ATOMS", "CMAP", "HBOND", "THOLE"}:
            if head in {"CMAP", "THOLE"}:
                unsupported.add(head)
            kind = None
            continue
        if kind is None or (kind == Kind.NONBONDED and head in _NONBONDED_OPTIONS):
            continue
        arity, sizes = _SHAPES[kind]
        types = tuple(fields[:arity])
        try:
            values = tuple(float(value.replace("D", "E")) for value in fields[arity:])
            if len(values) not in sizes or not all(isfinite(value) for value in values):
                raise ValueError("wrong number of columns or non-finite values")
            if kind in {Kind.PROPER, Kind.IMPROPER} and (
                not values[1].is_integer() or values[1] < 0
            ):
                raise ValueError("invalid torsion multiplicity")
            _validate_wildcards(kind, types)
        except ValueError as exc:
            raise AutoCGENFFError(f"{path}:{line_no}: invalid {kind} row: {exc}") from exc
        records = entries[kind].setdefault(_key(kind, types), [])
        record = ParameterRecord(types, values, line_no)
        if kind == Kind.PROPER:
            # Replace only the repeated multiplicity and keep surviving rows in file order.
            for index, previous in enumerate(records):
                if previous.values[1] == values[1]:
                    del records[index]
                    break
            records.append(record)
        elif kind == Kind.IMPROPER and "X" not in types:
            records.append(record)
        else:
            records[:] = [record]
    if not any(entries.values()):
        raise AutoCGENFFError(f"No supported parameters found in {path}")
    frozen = {
        kind: {key: tuple(rows) for key, rows in groups.items()} for kind, groups in entries.items()
    }
    return Parameters(path, frozen, tuple(sorted(unsupported)))
