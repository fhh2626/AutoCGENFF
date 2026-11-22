"""Read an additive CHARMM RTF and verify a caller-selected residue mapping."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from math import fsum, isclose, isfinite
from pathlib import Path

from rdkit import Chem

from .errors import AutoCGENFFError
from .molecule import _reject_query_bonds_and_allene_stereo, prepare_molecule
from .parameters import Types, charmm_lines

NamePath = tuple[str, ...]
_CHARGE_TOLERANCE = 1e-5
_SEARCH_LIMIT = 100_000


@dataclass(frozen=True, slots=True)
class TemplateAtom:
    """One named RTF atom with a type and partial charge in elementary charges."""

    name: str
    atom_type: str
    charge: float


@dataclass(frozen=True, slots=True)
class TemplateLonePair:
    """One collinear virtual lone-pair site declared in an RTF residue."""

    name: str
    host: str
    anchor: str
    distance: float
    scale: float = 0.0


@dataclass(slots=True)
class Residue:
    """Parsed residue topology; not a generated or patched CHARMM residue."""

    name: str
    charge: float
    line: int
    auto_angles: bool
    auto_propers: bool
    atoms: list[TemplateAtom] = field(default_factory=list)
    bonds: dict[tuple[str, str], int] = field(default_factory=dict)
    angles: list[NamePath] = field(default_factory=list)
    propers: list[NamePath] = field(default_factory=list)
    impropers: list[NamePath] = field(default_factory=list)
    lone_pairs: list[TemplateLonePair] = field(default_factory=list)
    unsupported: set[str] = field(default_factory=set)
    first_patch: str | None = None
    last_patch: str | None = None

    @property
    def patches(self) -> tuple[str, ...]:
        """Active patch names, in first/last order."""
        return tuple(patch for patch in (self.first_patch, self.last_patch) if patch is not None)


@dataclass(slots=True)
class Topology:
    """Parsed residue templates and explicit MASS element declarations."""

    source: Path
    residues: dict[str, Residue]
    elements: dict[str, str]
    masses: dict[str, float] = field(default_factory=dict)
    patch_names: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    """A named RTF assignment in the molecule's explicit-H atom order."""

    name: str
    source: str
    line: int
    atoms: tuple[TemplateAtom, ...]
    auto_angles: bool
    auto_propers: bool
    angles: tuple[tuple[int, ...], ...]
    propers: tuple[tuple[int, ...], ...]
    impropers: tuple[tuple[int, ...], ...]
    verified_reference_smiles: bool


_BOND_ORDERS = {"BOND": 1, "DOUB": 2, "DOUBLE": 2, "TRIP": 3, "TRIPLE": 3}
_DECLARED_TERMS = {
    "ANGL": ("angles", 3),
    "ANGLE": ("angles", 3),
    "THET": ("angles", 3),
    "THETA": ("angles", 3),
    "DIHE": ("propers", 4),
    "DIHEDRAL": ("propers", 4),
    "PHI": ("propers", 4),
    "IMPR": ("impropers", 4),
    "IMPROPER": ("impropers", 4),
    "IMPH": ("impropers", 4),
    "IMPHI": ("impropers", 4),
}
_IC_DIRECTIVES = {"IC", "BILD", "BUILD"}
_RTF_NOT_TERMS = _IC_DIRECTIVES | {
    "GROU",
    "GROUP",
    "DONO",
    "DONOR",
    "ACCE",
    "ACCEPTOR",
    "DECL",
    "DECLARE",
    "MASS",
}


def _is_starred_ic_atom(parts: list[str], tokens: list[str], continued: bool) -> bool:
    # * marks the third atom of an improper internal coordinate, not a title.
    return (
        len(parts) == 3
        and parts[0] in _IC_DIRECTIVES
        and len(tokens[0]) > 1
        and not tokens[0].startswith("**")
    )


def read_topology(path: Path) -> Topology:
    """Parse an RTF; defer unsupported directives to selection of the residue."""
    residues = {}
    patch_names = set()
    elements = {}
    masses = {}
    current = None
    auto_angles = auto_propers = False
    default_first = default_last = None
    for line_no, fields in charmm_lines(path, asterisk_row=_is_starred_ic_atom):
        head, *args = fields
        try:
            if head == "END":
                break
            if head == "MASS":
                mass = float(args[2])
                if not isfinite(mass) or mass < 0:
                    raise ValueError("invalid mass")
                masses[args[1]] = mass
                if len(args) >= 4:
                    elements[args[1]] = args[3].title()
            if head in {"AUTO", "AUTOGENERATE"}:
                auto_angles = any(arg.startswith("ANGL") for arg in args)
                auto_propers = any(arg.startswith("DIHE") for arg in args)
                continue
            if head in {"DEFA", "DEFAULT", "DEFAULTS"}:
                default_first, default_last = _patch_names(args, default_first, default_last)
                continue
            if head in {"PRES", "PRESIDUE"}:
                patch_names.add(args[0])
                current = None
                continue
            if head in {"RESI", "RESIDUE"}:
                name = args[0]
                if name in residues:
                    raise AutoCGENFFError(f"duplicate RESI {name}")
                current = Residue(
                    name,
                    float(args[1]),
                    line_no,
                    auto_angles,
                    auto_propers,
                    first_patch=default_first,
                    last_patch=default_last,
                )
                residues[name] = current
            elif current is not None:
                _read_directive(current, head, args)
        except (ValueError, IndexError) as exc:
            raise AutoCGENFFError(f"{path}:{line_no}: invalid RTF {head}: {exc}") from exc
    if not residues:
        raise AutoCGENFFError(f"No RESI definitions found in {path}")
    return Topology(path, residues, elements, masses, patch_names)


def _patch_names(
    args: list[str], first: str | None, last: str | None
) -> tuple[str | None, str | None]:
    for index in range(0, len(args), 2):
        end, name = args[index], args[index + 1]
        patch = None if name == "NONE" else name
        if end in {"FIRST", "FIRS"}:
            first = patch
        elif end == "LAST":
            last = patch
        else:
            raise ValueError(f"expected FIRST or LAST, not {end}")
    return first, last


def _read_directive(residue: Residue, head: str, args: list[str]) -> None:
    if head == "ATOM":
        residue.atoms.append(TemplateAtom(args[0], args[1], float(args[2])))
        if len(args) != 3:
            residue.unsupported.add("extended ATOM fields")
    elif head in _BOND_ORDERS:
        for pair in _chunks(args, 2):
            left, right = sorted(pair)
            key = (left, right)
            order = _BOND_ORDERS[head]
            if key in residue.bonds and residue.bonds[key] != order:
                residue.unsupported.add(f"conflicting bond orders {left}-{right}")
            residue.bonds[key] = order
    elif head in _DECLARED_TERMS:
        attribute, size = _DECLARED_TERMS[head]
        getattr(residue, attribute).extend(_chunks(args, size))
    elif head in {"PATC", "PATCH", "PATCHING"}:
        residue.first_patch, residue.last_patch = _patch_names(
            args, residue.first_patch, residue.last_patch
        )
    elif head in {"LONE", "LONEPAIR"}:
        lp = _parse_colinear_lonepair(args)
        if lp is None:
            residue.unsupported.add("LONEPAIR")
        else:
            residue.lone_pairs.append(lp)
    elif head not in _RTF_NOT_TERMS:
        residue.unsupported.add(head)


def _parse_colinear_lonepair(args: list[str]) -> TemplateLonePair | None:
    if len(args) < 6 or not args[0].startswith("COLI"):
        return None
    name, host, anchor = args[1], args[2], args[3]
    rest = args[4:]
    if len(rest) % 2:
        return None
    distance: float | None = None
    scale: float | None = None
    for idx in range(0, len(rest), 2):
        key, val = rest[idx], rest[idx + 1]
        num = float(val)
        if not isfinite(num):
            return None
        if key.startswith("DIST"):
            if distance is not None:
                return None
            distance = num
        elif key.startswith("SCAL"):
            if scale is not None:
                return None
            scale = num
        else:
            return None
    if distance is None or distance <= 0:
        return None
    return TemplateLonePair(name, host, anchor, distance, 0.0 if scale is None else scale)


def _chunks(args: list[str], size: int) -> list[NamePath]:
    if not args or len(args) % size:
        raise AutoCGENFFError(f"expected groups of {size} atom names")
    return [tuple(args[index : index + size]) for index in range(0, len(args), size)]


def match_template(
    molecule: Chem.Mol,
    topology: Topology,
    name: str,
    *,
    atom_map: Mapping[str, int] | None = None,
    reference_smiles: str | None = None,
) -> TemplateMatch:
    """Map a named template or raise AutoCGENFFError on uncertain correspondence.

    `atom_map`, if supplied, must map every RTF atom name to an explicit-H
    molecule index. RTF does not encode per-atom formal charges, stereochemistry
    or all aromatic bond orders; for those cases the caller must supply the
    independently established chemical identity of the named template as SMILES.
    """
    _reject_query_bonds_and_allene_stereo(molecule)
    residue = topology.residues.get(name.upper())
    if residue is None:
        raise AutoCGENFFError(f"Unknown RTF residue {name!r} in {topology.source}")
    if residue.unsupported or residue.patches or residue.lone_pairs:
        extra = {"LONEPAIR"} if residue.lone_pairs else set()
        features = ", ".join(sorted(residue.unsupported | set(residue.patches) | extra))
        raise AutoCGENFFError(
            f"RTF residue {residue.name} requires unsupported topology: {features}"
        )
    names = [atom.name for atom in residue.atoms]
    if len(set(names)) != len(names) or len(names) != molecule.GetNumAtoms():
        raise AutoCGENFFError(
            f"RTF residue {residue.name} has a different or invalid atom inventory"
        )
    total_charge = Chem.GetFormalCharge(molecule)
    if not isclose(
        residue.charge, total_charge, rel_tol=0, abs_tol=_CHARGE_TOLERANCE
    ) or not isclose(
        fsum(atom.charge for atom in residue.atoms),
        residue.charge,
        rel_tol=0,
        abs_tol=_CHARGE_TOLERANCE,
    ):
        raise AutoCGENFFError(f"RTF residue {residue.name} and molecule charge totals disagree")
    if molecule.GetStereoGroups():
        raise AutoCGENFFError("Enhanced stereochemistry is not supported for named templates")
    if _has_atropisomeric_stereo(molecule):
        raise AutoCGENFFError("Atropisomeric stereochemistry is not supported for named templates")
    if reference_smiles is not None:
        reference = prepare_molecule(reference_smiles)
        if reference.GetStereoGroups():
            raise AutoCGENFFError("Enhanced stereochemistry is not supported for template SMILES")
        if _has_atropisomeric_stereo(reference):
            raise AutoCGENFFError(
                "Atropisomeric stereochemistry is not supported for template SMILES"
            )
        if _identity(molecule) != _identity(reference):
            raise AutoCGENFFError("Molecule differs from the supplied template SMILES identity")
    elif _needs_reference_identity(molecule):
        raise AutoCGENFFError(
            "Aromatic, charged, radical, isotopic or stereochemical input requires template SMILES"
        )
    elements = _template_elements(residue, topology)
    neighbors = _template_neighbors(residue, names)
    has_identity = reference_smiles is not None
    if atom_map is not None:
        mapped = dict(atom_map)
        _validate_atom_map(molecule, mapped, elements, neighbors)
        return _match(residue, topology.source, mapped, has_identity)
    candidates = _mapping_candidates(molecule, names, elements, neighbors)
    mapping = None
    signature = None
    for proposed in _iter_mappings(molecule, neighbors, candidates):
        matched = _match(residue, topology.source, proposed, has_identity)
        proposed_signature = _mapping_signature(matched)
        if signature is None:
            mapping = matched
            signature = proposed_signature
        elif signature != proposed_signature:
            raise AutoCGENFFError(
                f"Ambiguous mapping to {residue.name}; provide an explicit atom map"
            )
    if mapping is None:
        raise AutoCGENFFError(
            f"Molecule does not match {residue.name} elements, connectivity and bond orders"
        )
    return mapping


def _has_atropisomeric_stereo(molecule: Chem.Mol) -> bool:
    # MolToSmiles omits axial bond stereo; older RDKit versions lack the enum members.
    return any(
        bond.GetStereo().name in {"STEREOATROPCW", "STEREOATROPCCW"} for bond in molecule.GetBonds()
    )


def _identity(molecule: Chem.Mol) -> str:
    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def _needs_reference_identity(molecule: Chem.Mol) -> bool:
    return any(
        atom.GetIsAromatic()
        or atom.GetFormalCharge()
        or atom.GetNumRadicalElectrons()
        or atom.GetIsotope()
        for atom in molecule.GetAtoms()
    ) or bool(Chem.FindPotentialStereo(molecule))


def _template_elements(residue: Residue, topology: Topology) -> dict[str, str]:
    result = {}
    for atom in residue.atoms:
        element = topology.elements.get(atom.atom_type)
        if not element:
            raise AutoCGENFFError(f"Missing element in MASS definition of type {atom.atom_type}")
        result[atom.name] = element
    return result


def _template_neighbors(residue: Residue, names: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {name: {} for name in names}
    for (left, right), order in residue.bonds.items():
        if left == right or left not in result or right not in result:
            raise AutoCGENFFError(
                f"External or invalid bond in residue {residue.name}: {left}-{right}"
            )
        result[left][right] = order
        result[right][left] = order
    return result


def _validate_atom_map(
    molecule: Chem.Mol,
    atom_map: dict[str, int],
    elements: dict[str, str],
    neighbors: dict[str, dict[str, int]],
) -> None:
    if any(type(index) is not int for index in atom_map.values()):
        raise AutoCGENFFError("Expected an integer atom index for each template atom name")
    if atom_map.keys() != elements.keys() or set(atom_map.values()) != set(range(len(elements))):
        raise AutoCGENFFError(
            "Template atom map must bijectively cover all names and molecule indices"
        )
    # Equal degrees and matching expected bonds also rule out any extra bonds.
    for name, index in atom_map.items():
        atom = molecule.GetAtomWithIdx(index)
        expected = neighbors[name]
        if (
            atom.GetSymbol() != elements[name]
            or atom.GetDegree() != len(expected)
            or any(
                not _bond_agrees(molecule.GetBondBetweenAtoms(index, atom_map[other]), order)
                for other, order in expected.items()
            )
        ):
            raise AutoCGENFFError(
                "Template atom map does not match molecule elements, connectivity and bond orders"
            )


def _mapping_candidates(
    molecule: Chem.Mol,
    names: list[str],
    elements: dict[str, str],
    neighbors: dict[str, dict[str, int]],
) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetSymbol() == elements[name] and atom.GetDegree() == len(neighbors[name])
        )
        for name in names
    }


def _iter_mappings(
    molecule: Chem.Mol,
    neighbors: dict[str, dict[str, int]],
    candidates: dict[str, tuple[int, ...]],
) -> Iterator[dict[str, int]]:
    mapped: dict[str, int] = {}
    used: set[int] = set()
    attempts = 0

    def search() -> Iterator[dict[str, int]]:
        nonlocal attempts
        if len(mapped) == len(candidates):
            yield dict(mapped)
            return
        name = min(
            (item for item in candidates if item not in mapped),
            key=lambda item: (
                -len(neighbors[item].keys() & mapped.keys()),
                len(candidates[item]),
                -len(neighbors[item]),
                item,
            ),
        )
        for index in candidates[name]:
            attempts += 1
            if attempts > _SEARCH_LIMIT:
                raise AutoCGENFFError(
                    "Template isomorphism search limit reached; provide an explicit atom map"
                )
            if index in used or not _agrees_with_mapped(molecule, index, neighbors[name], mapped):
                continue
            mapped[name] = index
            used.add(index)
            yield from search()
            used.remove(index)
            del mapped[name]

    yield from search()


def _agrees_with_mapped(
    molecule: Chem.Mol, index: int, expected: dict[str, int], mapped: dict[str, int]
) -> bool:
    return all(
        _bond_agrees(molecule.GetBondBetweenAtoms(index, other), expected.get(name))
        for name, other in mapped.items()
    )


def _bond_agrees(bond: Chem.Bond | None, order: int | None) -> bool:
    if bond is None:
        return order is None
    if bond.GetIsAromatic():
        # match_template mandates independent identity for aromatic input.
        return order in (1, 2)
    return bond.GetBondTypeAsDouble() == order


def _match(
    residue: Residue, source: Path, mapped: dict[str, int], has_identity: bool
) -> TemplateMatch:
    def translate(paths: list[NamePath]) -> tuple[tuple[int, ...], ...]:
        try:
            return tuple(tuple(mapped[name] for name in path) for path in paths)
        except KeyError as exc:
            raise AutoCGENFFError(
                f"External or unknown atom {exc} in {residue.name} topology"
            ) from exc

    return TemplateMatch(
        residue.name,
        str(source),
        residue.line,
        tuple(sorted(residue.atoms, key=lambda atom: mapped[atom.name])),
        residue.auto_angles,
        residue.auto_propers,
        () if residue.auto_angles else translate(residue.angles),
        () if residue.auto_propers else translate(residue.propers),
        translate(residue.impropers),
        has_identity,
    )


def improper_source_centers(topology: Topology) -> dict[Types, int]:
    """Index exact typed RTF impropers with unambiguous local centers at either end."""
    observed: dict[Types, set[int | None]] = {}
    for residue in topology.residues.values():
        types_by_name = {atom.name: atom.atom_type for atom in residue.atoms}
        if len(types_by_name) != len(residue.atoms):
            continue
        for path in residue.impropers:
            if not set(path) <= types_by_name.keys():
                continue  # Cannot establish a PRM type key for an external atom.
            types = tuple(types_by_name[name] for name in path)
            centers = [
                index
                for index in (0, 3)
                if all(
                    tuple(sorted((path[index], name))) in residue.bonds
                    for name in path
                    if name != path[index]
                )
            ]
            center = centers[0] if len(set(path)) == 4 and len(centers) == 1 else None
            observed.setdefault(types, set()).add(center)
            observed.setdefault(types[::-1], set()).add(None if center is None else 3 - center)
    return {
        types: next(iter(centers))
        for types, centers in observed.items()
        if len(centers) == 1 and None not in centers
    }


def _mapping_signature(
    match: TemplateMatch,
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    return (
        tuple((atom.atom_type, atom.charge) for atom in match.atoms),
        tuple(sorted(min(path, path[::-1]) for path in match.angles)),
        tuple(sorted(min(path, path[::-1]) for path in match.propers)),
        tuple(sorted(min(path, path[::-1]) for path in match.impropers)),
    )
