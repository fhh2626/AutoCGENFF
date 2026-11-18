"""Checked, changed-only CHARMM stream export for a single local-v5 residue."""

import re
from collections import Counter
from math import fsum, isclose, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from rdkit import Chem

from .assignment import AssignedTerm
from .errors import AutoCGENFFError
from .molecule import Kind, Term, inventory
from .parameters import Parameters, Types, read_parameters
from .topology import Residue, read_topology
from .v5_assignment import V5Assignment

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]{0,3}")
_SECTIONS = {
    Kind.BOND: "BONDS",
    Kind.ANGLE: "ANGLES",
    Kind.PROPER: "DIHEDRALS",
    Kind.IMPROPER: "IMPROPER",
}
_ORDERS = {1: "BOND", 2: "DOUBLE", 3: "TRIPLE"}
_TITLE = "* AutoCGENFF local-v5 ligand stream"


def _validate_names(result: V5Assignment, residue_name: str) -> None:
    if not _NAME.fullmatch(residue_name):
        raise AutoCGENFFError(
            "Residue name must be 1-4 ASCII letters/digits, starting with a letter"
        )
    used = set()
    for i, atom in enumerate(result.atoms):
        if atom.index != i or not _NAME.fullmatch(atom.name) or atom.name.upper() in used:
            raise AutoCGENFFError(
                f"Invalid, duplicate or unordered CHARMM atom name at {i}: {atom.name!r}"
            )
        used.add(atom.name.upper())


def _rtf_text(result: V5Assignment, residue_name: str) -> str:
    atoms = result.atoms
    charge = sum(a.formal_charge for a in atoms)
    lines = [
        "read rtf card append",
        "* AutoCGENFF local-v5 residue",
        "*",
        "36 1",
        "AUTO ANGLES DIHE",
        f"RESI {residue_name} {charge}",
        "GROUP",
    ]
    if result.charge_diagnostics.charges != tuple(a.partial_charge for a in atoms):
        raise AutoCGENFFError("Atom and increment charges disagree")
    for a in atoms:
        if a.partial_charge is None or not isfinite(a.partial_charge):
            raise AutoCGENFFError(f"Invalid charge at {a.index}")
        if not re.fullmatch(r"[A-Z][A-Z0-9]{0,7}", a.atom_type):
            raise AutoCGENFFError(f"Invalid CHARMM atom type {a.atom_type!r}")
        lines.append(f"ATOM {a.name} {a.atom_type} {a.partial_charge!r}")
    edges = set()
    for bond in result.bonds:
        i, j = bond.atoms
        if not 0 <= i < j < len(atoms) or (i, j) in edges or bond.order not in _ORDERS:
            raise AutoCGENFFError(f"Invalid covalent bond {bond}")
        edges.add((i, j))
        lines.append(f"{_ORDERS[bond.order]} {atoms[i].name} {atoms[j].name}")
    for t in result.terms:
        if any(type(i) is not int or not 0 <= i < len(atoms) for i in t.term.atoms):
            raise AutoCGENFFError(f"Invalid atom index in {t.term}")
        if tuple(atoms[i].atom_type for i in t.term.atoms) != t.target_types:
            raise AutoCGENFFError(f"Parameter target disagrees with atoms in {t.term}")
        if t.term.kind == Kind.IMPROPER:
            lines.append("IMPR " + " ".join(atoms[i].name for i in t.term.atoms))
    lines.extend(["PATCH FIRST NONE LAST NONE", "END", ""])
    return "\n".join(lines)


def _check_residue(res: Residue, result: V5Assignment) -> None:
    """Reconstruct the AUTO inventory from serialized names and connectivity."""
    atoms = result.atoms
    names = {a.name.upper(): a.index for a in atoms}
    if (
        [a.name for a in res.atoms] != list(names)
        or [a.atom_type for a in res.atoms] != [a.atom_type for a in atoms]
        or [a.charge for a in res.atoms] != [a.partial_charge for a in atoms]
        or res.unsupported
        or res.patches
        or not (res.auto_angles and res.auto_propers)
    ):
        raise AutoCGENFFError("Serialized topology changes atoms or generation directives")
    if res.charge != sum(a.formal_charge for a in atoms) or not isclose(
        fsum(a.charge for a in res.atoms), res.charge, rel_tol=0, abs_tol=1e-7
    ):
        raise AutoCGENFFError("Serialized partial charges do not sum to residue charge")
    graph = Chem.RWMol()
    for atom in atoms:
        rd_atom = Chem.Atom(atom.element)
        rd_atom.SetFormalCharge(atom.formal_charge)
        rd_atom.SetNoImplicit(True)
        graph.AddAtom(rd_atom)
    edges = {tuple(sorted((names[a], names[b]))): order for (a, b), order in res.bonds.items()}
    if edges != {b.atoms: b.order for b in result.bonds}:
        raise AutoCGENFFError("Serialized bond orders or connectivity changed")
    bond_types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for (i, j), order in edges.items():
        graph.AddBond(i, j, bond_types[order])
    if Chem.SanitizeMol(graph, catchErrors=True) or any(
        a.GetNumRadicalElectrons() for a in graph.GetAtoms()
    ):
        raise AutoCGENFFError("Serialized bond orders or valences are not a closed-shell ligand")
    if len(Chem.GetMolFrags(graph)) != 1:
        raise AutoCGENFFError("Stream export requires a single connected ligand residue")
    impropers = [tuple(names[n] for n in imp) for imp in res.impropers]
    if Counter(p[0] for p in impropers) != Counter(a.index for a in atoms if a.improper_center):
        raise AutoCGENFFError("Serialized improper centers differ from the typing rules")
    for center, *outer in impropers:
        neighbors = {a.GetIdx() for a in graph.GetAtomWithIdx(center).GetNeighbors()}
        if set(outer) != neighbors:
            raise AutoCGENFFError(f"Invalid serialized improper neighbors at center {center}")
    expected = Counter(inventory(graph, impropers))
    expected.update(Term(Kind.NONBONDED, (a.index,)) for a in atoms)
    if expected != Counter(t.term for t in result.terms):
        raise AutoCGENFFError("Serialized AUTO inventory differs from assigned paths")


def _delta(result: V5Assignment, base: Parameters) -> dict[tuple[Kind, Types], AssignedTerm]:
    groups = {}
    for t in result.terms:
        kind, target, match = t.term.kind, t.target_types, t.match
        if match is None or not match.records or len(match.records) != len(match.record_reversed):
            raise AutoCGENFFError(f"Incomplete parameter selection for {t.term}")
        native = base.match(kind, target)
        if t.status in {"exact", "wildcard"}:
            if native != match or t.status != ("wildcard" if native.wildcard else "exact"):
                raise AutoCGENFFError(f"Base parameter changed at {kind} {target}")
            continue
        if t.status != "analogy" or native is not None or kind not in _SECTIONS:
            raise AutoCGENFFError(f"Refusing incompatible parameter override for {kind} {target}")
        source = base.match(kind, match.records[0].types)
        if source is None or source.records != match.records or source.wildcard:
            raise AutoCGENFFError(f"Analogy source no longer matches base for {kind} {target}")
        orientations = {
            r.types[::-1] if rev else r.types
            for r, rev in zip(match.records, match.record_reversed, strict=True)
        }
        if len(orientations) != 1:
            raise AutoCGENFFError(f"Inconsistent row orientations for {kind} {target}")
        key = kind, min(target, target[::-1])
        other = groups.get(key)
        if other is not None and tuple(r.values for r in other.match.records) != tuple(
            r.values for r in match.records
        ):
            raise AutoCGENFFError(f"Conflicting analogy parameters for {key}")
        groups[key] = t
    return groups


def _prm_text(groups: dict[tuple[Kind, Types], AssignedTerm]) -> str:
    lines = ["read param card flex append", "* AutoCGENFF local-v5 analogy additions", "*"]
    for kind, section in _SECTIONS.items():
        rows = [(target, t) for (k, target), t in groups.items() if k == kind]
        if not rows:
            continue
        lines.append(section)
        for target, t in rows:
            for row in t.match.records:
                # A full reversal preserves the signed dihedral. Write every Fourier
                # component contiguously in one orientation; never permute an improper.
                values = [
                    str(int(v)) if i == 1 and kind in {Kind.PROPER, Kind.IMPROPER} else repr(v)
                    for i, v in enumerate(row.values)
                ]
                lines.append(
                    " ".join((*target, *values))
                    + f" ! from base line {row.line}, penalty={t.penalty}"
                )
    return "\n".join([*lines, "END", ""])


def write_v5_stream(result: V5Assignment, path: Path, *, residue_name: str = "LIG") -> None:
    """Atomically write a checked additive .str to accompany the original local RTF/PRM.

    Raises AutoCGENFFError on unresolved, lossy or conflicting output; OSError on I/O
    failure. Reopens the base files; only replaces files generated by this writer.
    No PSF, coordinates or assertion of force-field quality is produced.
    """
    path = Path(path)
    if not result.is_complete:
        raise AutoCGENFFError(
            f"Cannot export incomplete local-v5 assignment: {'; '.join(result.issues)}"
        )
    if result.omitted_linear_propers:
        raise AutoCGENFFError("AUTO would regenerate omitted linear torsions; stream not written")
    if path.suffix.lower() != ".str" or path.is_symlink():
        raise AutoCGENFFError("Stream destination must be a regular .str path, not a symlink")
    if path.exists():
        if not path.is_file():
            raise AutoCGENFFError(f"Not a regular stream file: {path}")
        marker = _TITLE.encode("ascii")
        with path.open("rb") as handle:
            if handle.readline(len(marker) + 2) not in (marker + b"\n", marker + b"\r\n"):
                raise AutoCGENFFError("Refusing to overwrite a stream not generated by AutoCGENFF")
    _validate_names(result, residue_name)
    base_rtf = read_topology(Path(result.topology_source))
    if residue_name.upper() in base_rtf.residues or residue_name.upper() in base_rtf.patch_names:
        raise AutoCGENFFError(f"Residue name {residue_name} collides with the base RTF")
    for a in result.atoms:
        if (
            a.mass is None
            or not isfinite(a.mass)
            or a.mass <= 0
            or base_rtf.masses.get(a.atom_type) != a.mass
            or base_rtf.elements.get(a.atom_type, a.element) != a.element
        ):
            raise AutoCGENFFError(f"Incompatible base MASS for {a.atom_type}")
    base = read_parameters(Path(result.parameter_source))
    if (
        base.unsupported_sections
        or base.nbfix_for(tuple(a.atom_type for a in result.atoms)) != result.nbfix
    ):
        raise AutoCGENFFError("Incompatible base PRM sections or NBFIX overrides")
    rtf = _rtf_text(result, residue_name)
    groups = _delta(result, base)
    with TemporaryDirectory(dir=path.parent) as scratch:
        directory = Path(scratch)
        topology = directory / "ligand.rtf"
        topology.write_text(rtf, encoding="ascii")
        _check_residue(read_topology(topology).residues[residue_name.upper()], result)
        prm = ""
        if groups:
            params = directory / "delta.prm"
            params.write_text(_prm_text(groups), encoding="ascii")
            delta = read_parameters(params)
            for (kind, target), term in groups.items():
                loaded = delta.match(kind, target)
                if loaded is None or tuple(r.values for r in loaded.records) != tuple(
                    r.values for r in term.match.records
                ):
                    raise AutoCGENFFError(f"Lossy serialized analogy {kind} {target}")
            prm = params.read_text(encoding="ascii")
        stream = directory / "checked.str"
        stream.write_text(
            f"{_TITLE}\n*\n\n" + topology.read_text(encoding="ascii") + prm + "RETURN\n",
            encoding="ascii",
        )
        stream.replace(path)
