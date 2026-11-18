"""Command-line JSON diagnostics for a single molecule."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from math import isfinite
from pathlib import Path

from rdkit import rdBase

from .analogy import BondGroup, PenaltyModel
from .assignment import AssignmentOptions, parameterize
from .charges import ChargeModel
from .errors import AutoCGENFFError
from .parameters import read_parameters
from .topology import read_topology
from .typing import TypingRule
from .v5 import type_v5
from .v5_assignment import parameterize_v5
from .v5_stream import write_v5_stream


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise AutoCGENFFError(message)


def _add_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", help="single molecule as SMILES")
    source.add_argument("--input", type=Path, help="single molecule in SDF or MOL2")


def _parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(
        description=__doc__,
        epilog="Local workflows: autocgenff assign-v5 --help (ligand stream and audit), autocgenff type-v5 --help (typing only).",
    )
    _add_source(parser)
    parser.add_argument("--prm", type=Path, required=True, help="local CHARMM parameter file")
    parser.add_argument("--rtf", type=Path, help="local CHARMM topology file")
    parser.add_argument("--template", help="explicitly identified RTF residue name")
    parser.add_argument("--template-smiles", help="independently known identity of that residue")
    parser.add_argument(
        "--atom-map", type=Path, help="JSON object from RTF names to zero-based atom indices"
    )
    parser.add_argument("--rules", type=Path, help="JSON list of TypingRule keyword objects")
    parser.add_argument("--no-builtin-rules", action="store_true")
    parser.add_argument("--penalties", type=Path, help="JSON directional substitution model")
    parser.add_argument("--increments", type=Path, help="JSON charge increment model")
    parser.add_argument(
        "--formal-baseline", type=Path, help="JSON list of charges in explicit atom order"
    )
    impropers = parser.add_mutually_exclusive_group()
    impropers.add_argument(
        "--impropers", type=Path, help="JSON array of ordered zero-based quartets"
    )
    impropers.add_argument(
        "--no-impropers", action="store_true", help="assert no impropers are required"
    )
    return parser


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise AutoCGENFFError(f"Duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique_object)


def _object(value: object, allowed: set[str] | None = None) -> dict[str, object]:
    if not isinstance(value, dict) or (allowed is not None and set(value) - allowed):
        raise AutoCGENFFError(f"Expected JSON object with allowed keys {allowed}")
    return value


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AutoCGENFFError("Expected a JSON array")
    return value


def _label(value: object) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise AutoCGENFFError("Expected a nonempty CHARMM type or name")
    return value.upper()


def _index(value: object) -> int:
    if type(value) is not int or value < 0:
        raise AutoCGENFFError("Expected a nonnegative integer atom index")
    return value


def _number(value: object) -> float:
    if type(value) not in (int, float):
        raise AutoCGENFFError("Expected a JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise AutoCGENFFError("Expected a finite JSON number in float range") from exc
    if not isfinite(number):
        raise AutoCGENFFError("Expected a finite JSON number in float range")
    return number


def _read_increments(path: Path) -> ChargeModel:
    data = _object(_read_json(path), {"entries", "max_order", "torsion_cutoff"})
    increments = {}
    for row in _array(data.get("entries")):
        entry = _object(row, {"types", "values"})
        types = tuple(_label(value) for value in _array(entry.get("types")))
        values = tuple(_number(value) for value in _array(entry.get("values")))
        if types in increments:
            raise AutoCGENFFError(f"Duplicate increment key {types}")
        increments[types] = values
    return ChargeModel(
        increments, _index(data.get("max_order", 4)), _number(data.get("torsion_cutoff", 50))
    )


def _matrix(value: object) -> dict[tuple[str, str], float]:
    result = {}
    for entry in _array(value):
        row = _array(entry)
        if len(row) != 3:
            raise AutoCGENFFError("Penalty rows must have target type, source type, and penalty")
        pair = (_label(row[0]), _label(row[1]))
        if pair in result:
            raise AutoCGENFFError(f"Duplicate substitution {pair}")
        result[pair] = _number(row[2])
    return result


def _read_penalties(path: Path) -> PenaltyModel:
    data = _object(_read_json(path), {"bonded", "nonbonded", "bond_groups", "max_penalty"})
    groups = []
    for row in _array(data.get("bond_groups", [])):
        entry = _object(row, {"name", "members", "penalty"})
        groups.append(
            BondGroup(
                _label(entry.get("name")),
                frozenset(_label(item) for item in _array(entry.get("members"))),
                _number(entry.get("penalty")),
            )
        )
    maximum = data.get("max_penalty")
    return PenaltyModel(
        _matrix(data.get("bonded", [])),
        _matrix(data.get("nonbonded", [])),
        tuple(groups),
        None if maximum is None else _number(maximum),
    )


def _options(args: argparse.Namespace) -> AssignmentOptions:
    atom_map = None
    if args.atom_map is not None:
        atom_map = {}
        for name, index in _object(_read_json(args.atom_map)).items():
            label = _label(name)
            if label in atom_map:
                raise AutoCGENFFError(f"Duplicate template atom name after normalization: {label}")
            atom_map[label] = _index(index)
    impropers = () if args.no_impropers else None
    if args.impropers is not None:
        impropers = tuple(
            tuple(_index(index) for index in _array(row))
            for row in _array(_read_json(args.impropers))
        )
    return AssignmentOptions(
        rules=()
        if args.rules is None
        else tuple(
            TypingRule(**_object(row, {"name", "smarts", "atom_type", "priority", "target"}))
            for row in _array(_read_json(args.rules))
        ),
        include_builtin=not args.no_builtin_rules,
        penalties=None if args.penalties is None else _read_penalties(args.penalties),
        charge_model=None if args.increments is None else _read_increments(args.increments),
        formal_baseline=None
        if args.formal_baseline is None
        else tuple(_number(value) for value in _array(_read_json(args.formal_baseline))),
        impropers=impropers,
        topology=None if args.rtf is None else read_topology(args.rtf),
        template=args.template,
        template_smiles=args.template_smiles,
        atom_map=atom_map,
    )


def _assign_v5(arguments: Sequence[str]) -> int:
    parser = _JSONArgumentParser(
        prog="autocgenff assign-v5",
        description="Parameterize one ligand using local additive v5 inputs; write JSON audit to stdout.",
    )
    _add_source(parser)
    parser.add_argument(
        "--force-field",
        type=Path,
        required=True,
        help="directory containing the base CGenFF RTF and PRM",
    )
    parser.add_argument(
        "--increments",
        type=Path,
        help="native increment PRM (default: conventional filename beside base PRM)",
    )
    parser.add_argument(
        "--stream", type=Path, help="write a checked CHARMM .str only on complete coverage"
    )
    parser.add_argument("--residue", default="LIG", help="new stream residue name (default LIG)")
    args = parser.parse_args(arguments)
    with rdBase.BlockLogs():
        result = parameterize_v5(
            args.input if args.smiles is None else args.smiles,
            args.force_field,
            increments_path=args.increments,
        )
    document = {**asdict(result), "is_complete": result.is_complete, "stream_written": None}
    if args.stream is not None and result.is_complete:
        try:
            write_v5_stream(result, args.stream, residue_name=args.residue)
        except (AutoCGENFFError, OSError) as exc:
            document["stream_error"] = str(exc)
            print(json.dumps(document, indent=2, allow_nan=False))
            raise
        document["stream_written"] = {"path": str(args.stream), "residue": args.residue}
    print(json.dumps(document, indent=2, allow_nan=False))
    return 0 if result.is_complete else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Write JSON to stdout; return 0 complete, 1 incomplete, 2 invalid input."""
    try:
        arguments = list(sys.argv[1:] if argv is None else argv)
        if arguments[:1] == ["assign-v5"]:
            return _assign_v5(arguments[1:])
        if arguments[:1] == ["type-v5"]:
            parser = _JSONArgumentParser(
                prog="autocgenff type-v5",
                description="Type atoms only using the bundled local-v5 profile.",
            )
            _add_source(parser)
            args = parser.parse_args(arguments[1:])
            with rdBase.BlockLogs():
                typed = type_v5(args.input if args.smiles is None else args.smiles)
            print(json.dumps({**asdict(typed), "is_complete": typed.is_complete}, indent=2))
            return 0 if typed.is_complete else 1
        args = _parser().parse_args(arguments)
        # Native RDKit logs would otherwise corrupt the CLI's JSON stderr channel.
        with rdBase.BlockLogs():
            options = _options(args)
            parameters = read_parameters(args.prm)
            result = parameterize(
                args.input if args.smiles is None else args.smiles, parameters, options
            )
        document = asdict(result)
        document["is_complete"] = result.is_complete
        print(json.dumps(document, indent=2, allow_nan=False))
        return 0 if result.is_complete else 1
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
