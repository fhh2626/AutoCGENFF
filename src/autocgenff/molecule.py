"""RDKit molecule preparation and explicit-hydrogen bonded inventory."""

import re
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations, product
from pathlib import Path
from typing import BinaryIO

from rdkit import Chem

from .errors import AutoCGENFFError


class Kind(StrEnum):
    """Supported parameter and path kinds."""

    BOND = "bond"
    ANGLE = "angle"
    PROPER = "proper"
    IMPROPER = "improper"
    NONBONDED = "nonbonded"
    NBFIX = "nbfix"


@dataclass(frozen=True, slots=True)
class Term:
    """An ordered tuple of zero-based atom indices, classified by kind."""

    kind: Kind
    atoms: tuple[int, ...]


def prepare_molecule(source: str | Path | Chem.Mol) -> Chem.Mol:
    """Return a sanitized independent explicit-H molecule, preserving input indices.

    A string is interpreted as SMILES. A Path must name a single-molecule SDF
    or MOL2. Existing H atoms stay in place; new H atoms are appended by RDKit.
    Raises AutoCGENFFError on empty/invalid or unsupported input.
    """
    try:
        if isinstance(source, Chem.Mol):
            molecule = Chem.Mol(source)
        elif isinstance(source, Path):
            molecule = _read_molecule(source)
        else:
            lines = [line.strip() for line in source.splitlines() if line.strip()]
            if len(lines) != 1 or any(char.isspace() or char == "|" for char in lines[0]):
                raise AutoCGENFFError(
                    "Expected one plain SMILES; CXSMILES and names are unsupported"
                )
            options = Chem.SmilesParserParams()
            options.allowCXSMILES = False
            options.removeHs = False
            options.parseName = False
            options.sanitize = False  # Inspect explicit allene stereo before RDKit clears it.
            molecule = Chem.MolFromSmiles(lines[0], options)
        if molecule is None or not molecule.GetNumAtoms():
            raise AutoCGENFFError("Input does not contain a valid nonempty molecule")
        _reject_query_bonds_and_allene_stereo(molecule)
        if any(atom.GetAtomicNum() == 0 or atom.HasQuery() for atom in molecule.GetAtoms()):
            raise AutoCGENFFError("Dummy and query atoms are not supported")
        Chem.SanitizeMol(molecule)
        # Some RDKit molecules store valid E/Z only on the double bond. Restore the
        # neighboring directions needed by RDKit's clean stereochemistry assignment.
        Chem.SetDoubleBondNeighborDirections(molecule)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        if any(bond.GetBondType() not in _BOND_TYPES for bond in molecule.GetBonds()):
            raise AutoCGENFFError("Only single, double, triple and aromatic bonds are supported")
        return Chem.AddHs(molecule)
    except AutoCGENFFError:
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        raise AutoCGENFFError(f"Cannot read or sanitize molecule: {exc}") from exc


_SDF_FIELD = re.compile(rb">[^<>]*<[^<>]*[^<>\s][^<>]*>[^<>]*")
_MOL2_HEADER = "@<TRIPOS>"
_BOND_TYPES = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}


def reject_ligand_water(molecule: Chem.Mol) -> None:
    """Reject neutral water components of a prepared, explicit-H ligand graph."""
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetFormalCharge() or atom.GetDegree() != 2:
            continue
        if all(
            other.GetAtomicNum() == 1 and other.GetFormalCharge() == 0 and other.GetDegree() == 1
            for other in atom.GetNeighbors()
        ) and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()):
            raise AutoCGENFFError(
                f"Neutral water component at oxygen {atom.GetIdx()}; use toppar_water_ions.str "
                "separately, not ligand parameterization"
            )


def _reject_query_bonds_and_allene_stereo(molecule: Chem.Mol) -> None:
    if any(bond.HasQuery() for bond in molecule.GetBonds()):
        raise AutoCGENFFError("Query bonds are not supported")
    if any(atom.GetChiralTag() == Chem.ChiralType.CHI_ALLENE for atom in molecule.GetAtoms()):
        raise AutoCGENFFError("Explicit allene stereochemistry is not supported")


def _read_molecule(path: Path) -> Chem.Mol | None:
    match path.suffix.lower():
        case ".mol2":
            return _read_mol2(path)
        case ".sdf":
            with path.open("rb") as handle:
                _validate_sdf_record(handle, path)
                handle.seek(0)
                return next(iter(Chem.ForwardSDMolSupplier(handle, removeHs=False)), None)
        case _:
            raise AutoCGENFFError(f"Expected a .mol2 or .sdf file, not {path}")


def _read_mol2(path: Path) -> Chem.Mol | None:
    block = path.read_text(encoding="utf-8")
    lines = block.splitlines()
    start = _validate_mol2_record(lines, path)
    title = lines[start + 1]
    if title.startswith(_MOL2_HEADER):
        # RDKit can rescan a marker-shaped title as a header. It is only metadata.
        lines[start + 1] = "MOL2 molecule"
        block = "\n".join(lines) + "\n"
    molecule = Chem.MolFromMol2Block(block, removeHs=False)
    if molecule is not None:
        expected = tuple(map(int, lines[start + 2].split()[:2]))
        if expected != (molecule.GetNumAtoms(), molecule.GetNumBonds()):
            raise AutoCGENFFError(f"RDKit would discard MOL2 atoms or bonds in {path}")
        molecule.SetProp("_Name", title)
    return molecule


def _validate_mol2_record(lines: Sequence[str], path: Path) -> int:
    header = f"{_MOL2_HEADER}MOLECULE"
    start = next((index for index, line in enumerate(lines) if line.rstrip() == header), None)
    if start is None or start + 2 >= len(lines):
        raise AutoCGENFFError(f"Expected one MOL2 molecule in {path}")
    # The line immediately after the molecule marker is a free-form title.
    count_fields = lines[start + 2].split()
    try:
        atom_count, bond_count = map(int, count_fields[:2])
    except ValueError as exc:
        raise AutoCGENFFError(f"Invalid MOL2 atom or bond counts in {path}") from exc
    counts = {"ATOM": 0, "BOND": 0}
    seen = set()
    section = ""
    known_markers = tuple(f"{_MOL2_HEADER}{name}" for name in ("MOLECULE", *counts))
    for index, raw in enumerate(lines):
        if index == start + 1:
            continue
        line = raw.rstrip()
        if line.startswith(known_markers) and line not in known_markers:
            raise AutoCGENFFError(f"Invalid MOL2 section marker at {path}:{index + 1}")
        if index < start + 3:
            continue
        if line.startswith(_MOL2_HEADER):
            section = line.removeprefix(_MOL2_HEADER)
            if section == "MOLECULE":
                raise AutoCGENFFError(f"Expected one MOL2 molecule in {path}")
            if section in counts:
                if section in seen:
                    raise AutoCGENFFError(f"Duplicate MOL2 {section} section in {path}")
                seen.add(section)
        elif section in counts and line and not line.lstrip().startswith("#"):
            counts[section] += 1
    if (counts["ATOM"], counts["BOND"]) != (atom_count, bond_count) or atom_count <= 0:
        raise AutoCGENFFError(f"MOL2 ATOM/BOND rows disagree with declared counts in {path}")
    return start


def _validate_sdf_record(handle: BinaryIO, path: Path) -> None:
    # The title, program line and comment are not delimiters even when they contain one.
    for _ in range(3):
        handle.readline()
    for line in handle:
        if line.rstrip() == b"M  END":
            break
    else:
        raise AutoCGENFFError(f"Expected one SDF record in {path}: missing M  END")

    # RDKit can yield the first molecule even when unrecognized data follows M  END.
    # Only SD data fields, a record separator and whitespace may follow the mol block.
    in_data = False
    for line in handle:
        if line.rstrip() == b"$$$$":
            if in_data:
                break  # A blank line must end a data item before the separator.
            if any(chunk.strip() for chunk in iter(lambda: handle.read(65536), b"")):
                break
            return
        if in_data:
            if not line.rstrip(b"\r\n"):
                in_data = False
        elif _SDF_FIELD.fullmatch(line.strip()):
            in_data = True
        elif line.strip():
            break
    else:
        # Preserve acceptance of a single mol block without a final separator.
        return
    raise AutoCGENFFError(f"Expected one SDF record in {path}: invalid or additional content")


def inventory(
    molecule: Chem.Mol,
    impropers: Sequence[tuple[int, ...]] = (),
    *,
    explicit_angles: Sequence[tuple[int, ...]] = (),
    explicit_propers: Sequence[tuple[int, ...]] = (),
    auto_angles: bool = True,
    auto_propers: bool = True,
) -> tuple[Term, ...]:
    """Enumerate graph terms plus ordered declared impropers without mutating input.

    Explicit impropers must have four distinct atoms and a central atom at
    position 0 (or at position 3 for a reversed declaration). All declared
    paths must use integer atom indices, not booleans or floats.
    Raises AutoCGENFFError for invalid declared terms.
    """
    neighbors = [sorted(n.GetIdx() for n in atom.GetNeighbors()) for atom in molecule.GetAtoms()]
    bonds = sorted(
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in molecule.GetBonds()
    )
    angles = set()
    propers = set()
    if auto_angles:
        for center, adjacent in enumerate(neighbors):
            angles.update((left, center, right) for left, right in combinations(adjacent, 2))
    if auto_propers:
        for first, second in bonds:
            for left, right in product(neighbors[first], neighbors[second]):
                if len({left, first, second, right}) == 4:
                    path = (left, first, second, right)
                    propers.add(min(path, path[::-1]))
    for declared, size, paths in ((explicit_angles, 3, angles), (explicit_propers, 4, propers)):
        for raw_path in declared:
            path = _coerce_path(raw_path)
            if not _valid_indices(path, size, len(neighbors)) or any(
                right not in neighbors[left] for left, right in zip(path, path[1:], strict=False)
            ):
                raise AutoCGENFFError(f"Invalid declared covalent path: {path}")
            paths.add(min(path, path[::-1]))
    ordered_impropers = []
    seen: set[tuple[int, frozenset[int]] | tuple[int, ...]] = set()
    for raw_path in impropers:
        path = _coerce_path(raw_path)
        centers = (
            [
                center
                for center, adjacent in ((path[0], path[1:]), (path[-1], path[:-1]))
                if all(index in neighbors[center] for index in adjacent)
            ]
            if _valid_indices(path, 4, len(neighbors))
            else []
        )
        if not centers:
            raise AutoCGENFFError(f"Invalid declared improper (central atom first or last): {path}")
        key = (
            (centers[0], frozenset(path) - {centers[0]})
            if len(centers) == 1
            else min(path, path[::-1])
        )
        if key in seen:
            raise AutoCGENFFError(f"Duplicate declared improper: {path}")
        seen.add(key)
        ordered_impropers.append(path)
    result = []
    for kind, paths in (
        (Kind.BOND, bonds),
        (Kind.ANGLE, sorted(angles)),
        (Kind.PROPER, sorted(propers)),
        (Kind.IMPROPER, ordered_impropers),
    ):
        result.extend(Term(kind, path) for path in paths)
    return tuple(result)


def _coerce_path(path: object) -> tuple[int, ...]:
    if isinstance(path, (str, bytes, Mapping, AbstractSet)) or not isinstance(path, Iterable):
        raise AutoCGENFFError(f"Expected a sequence of atom indices, not {path!r}")
    return tuple(path)


def _valid_indices(path: tuple[int, ...], size: int, atom_count: int) -> bool:
    return (
        len(path) == size
        and all(type(index) is int and 0 <= index < atom_count for index in path)
        and len(set(path)) == size
    )
