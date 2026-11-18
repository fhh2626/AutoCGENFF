"""Optional integration: independent PSF generation and independent increment reader.

No engine, upstream package or force-field data are needed by the runtime.
"""

import os
import subprocess
from collections import Counter
from math import fsum
from pathlib import Path

import pytest

from autocgenff import V5ForceField, parameterize_v5, write_v5_stream
from autocgenff.molecule import Kind

# Named correspondence; never passed to inference. Includes two RTF-supported
# nitroalkanes for which an NG2O1 center must generate an ordered improper.
EXTRA = [
    ("NIME", "C[N+](=O)[O-]", "C1 N2 O21 O22 H11 H12 H13"),
    ("NIPR", "CCC[N+](=O)[O-]", "C3 C2 C1 N1 O11 O12 H31 H32 H33 H21 H22 H11 H12"),
    ("DMSO", "CS(C)=O", "C3 S2 C7 O1 H4 H5 H6 H8 H9 H10"),
    ("FURA", "c1ccco1", "C1 C2 C3 C4 O5 H1 H2 H3 H4"),
    ("FETH", "CCF", "C1 C2 F21 H11 H12 H13 H22 H23"),
    ("BORO", "CB(O)O", "C1 B1 O1 O2 H11 H12 H13 HO1 HO2"),
    ("SM169", "CNC(=S)N", "CD NE CZ SZ NH HD1 HD2 HD3 HE HH1 HH2"),
]


@pytest.fixture(scope="module")
def force_field():
    directory = Path(
        os.environ.get("AUTOCGENFF_V5_DIR", Path(__file__).resolve().parents[1] / "force_fields")
    )
    inc = Path(
        os.environ.get("AUTOCGENFF_V5_INCREMENTS", directory / "par_all36_cgenff_increments.prm")
    )
    if not inc.is_file() or not (directory / "top_all36_cgenff.rtf").is_file():
        pytest.skip("requires local v5 inputs")
    return V5ForceField.from_directory(directory, increments_path=inc)


@pytest.fixture(scope="module")
def independent_table(force_field):
    inc = pytest.importorskip("ligandff.charges.increments")
    return inc.IncrementTable.from_prm(force_field.increments_path)


@pytest.mark.parametrize("name,smiles,names", EXTRA)
def test_named_increment_and_path_crosscheck(name, smiles, names, force_field, independent_table):
    inc = pytest.importorskip("ligandff.charges.increments")
    r = parameterize_v5(smiles, force_field)
    assert r.is_complete and all(t.status == "exact" for t in r.terms)
    ref = force_field.topology.residues[name]
    labels = {a.name: a.atom_type for a in ref.atoms}
    names = names.split()
    types = [labels[n] for n in names]
    assert [a.atom_type for a in r.atoms] == types
    bonds = [(names.index(a), names.index(b)) for a, b in ref.bonds]
    charges = inc.assign_charges(types, bonds, [0] * len(names), independent_table)
    assert not charges.missing
    assert [a.partial_charge for a in r.atoms] == pytest.approx(charges.charges)
    paths = inc.parameter_instances(len(names), bonds)
    assert Counter(min(p, p[::-1]) for p in paths) == Counter(
        t.term.atoms for t in r.terms if t.term.kind in {Kind.BOND, Kind.ANGLE, Kind.PROPER}
    )
    assert [t.target_types for t in r.terms if t.term.kind == Kind.IMPROPER] == [
        tuple(labels[n] for n in p) for p in ref.impropers
    ]


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "C[NH3+]",
        "CC(=O)[O-]",
        "CC(=O)N",
        "c1ccncc1",
        "O=C=O",
        "CC(C)(C)CO",
        *(s for _, s, _ in EXTRA),
    ],
)
def test_psfgen_load_actual_stream_and_compare_entire_psf(smiles, force_field, tmp_path):
    exe = os.environ.get("AUTOCGENFF_PSFGEN")
    if not exe:
        pytest.skip("set AUTOCGENFF_PSFGEN to a local psfgen executable")
    result = parameterize_v5(smiles, force_field)
    stream = tmp_path / "ligand.str"
    write_v5_stream(result, stream)
    base = Path(os.path.relpath(result.topology_source, tmp_path)).as_posix()
    script = f"topology {{{base}}}\ntopology ligand.str\nsegment L {{ residue 1 LIG }}\nwritepsf x-plor ligand.psf\nexit\n"
    (tmp_path / "generate.tcl").write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [exe, "generate.tcl"], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    path = tmp_path / "ligand.psf"
    assert proc.returncode == 0 and path.is_file(), proc.stdout + proc.stderr
    lines = path.read_text(encoding="ascii").splitlines()
    sections = {
        line.split("!")[1].split(":")[0].strip(): (int(line.split()[0]), i + 1)
        for i, line in enumerate(lines)
        if "!" in line
    }
    n, start = sections["NATOM"]
    assert n == len(result.atoms)
    parsed = [line.split() for line in lines[start : start + n]]
    for atom, row in zip(result.atoms, parsed, strict=True):
        assert int(row[0]) == atom.index + 1 and row[3:6] == ["LIG", atom.name, atom.atom_type]
        assert float(row[6]) == pytest.approx(atom.partial_charge, abs=5.1e-7)
        assert float(row[7]) == pytest.approx(atom.mass, abs=5.1e-5)
    assert fsum(float(row[6]) for row in parsed) == pytest.approx(
        sum(a.formal_charge for a in result.atoms), abs=1e-5
    )
    for kind, key, arity in [
        (Kind.BOND, "NBOND", 2),
        (Kind.ANGLE, "NTHETA", 3),
        (Kind.PROPER, "NPHI", 4),
        (Kind.IMPROPER, "NIMPHI", 4),
    ]:
        count, start = sections[key]
        values = []
        for line in lines[start:]:
            if not line.strip():
                break
            values.extend(int(item) - 1 for item in line.split())
        assert len(values) == count * arity
        psf_paths = [tuple(values[i : i + arity]) for i in range(0, len(values), arity)]
        expected = [t.term.atoms for t in result.terms if t.term.kind == kind]
        if kind != Kind.IMPROPER:
            psf_paths = [min(p, p[::-1]) for p in psf_paths]
        assert Counter(psf_paths) == Counter(expected)


def test_independent_parser_reads_changed_only_prm(force_field, tmp_path):
    reader = pytest.importorskip("ligandff.io.prm")
    r = parameterize_v5("CC(C)(C)CO", force_field)
    path = tmp_path / "neo.str"
    write_v5_stream(r, path)
    prm = tmp_path / "delta.prm"
    prm.write_text(path.read_text().split("read param card flex append\n")[1], encoding="ascii")
    parsed = reader.read_parameters(prm)
    term = next(t for t in r.terms if t.status == "analogy")
    (record,) = parsed.bonded["dihedral"].values()
    assert record.types in (term.target_types, term.target_types[::-1])
    assert record.values == tuple(row.values for row in term.match.records)
    assert not any(parsed.bonded[k] for k in ("bond", "angle", "improper"))
    assert not parsed.nonbonded and not parsed.nbfix
