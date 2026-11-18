"""Original synthetic force field: test the CLI and stream without any licensed inputs."""

import json
from dataclasses import replace

import pytest
from rdkit import Chem

from autocgenff import (
    AutoCGENFFError,
    V5ForceField,
    parameterize_v5,
    read_topology,
    write_v5_stream,
)
from autocgenff.cli import main
from autocgenff.molecule import Kind
from autocgenff.parameters import read_parameters


@pytest.fixture
def bundle(tmp_path):
    # HGA3 -> HGA2 has one observed outer-position substitution on a bond;
    # the target proper is absent. The source Fourier family has BOTH orientations.
    (tmp_path / "top_all36_cgenff.rtf").write_text(
        "MASS -1 HGA3 1 H\nMASS -1 CG331 12 C\nRESI TAKN 0\nPRES PTCH 0\nEND\n", encoding="ascii"
    )
    (tmp_path / "par_all36_cgenff.prm").write_text(
        """BONDS
CG331 CG331 10 1
CG331 HGA2 11 1
CG331 HGA3 12 1 ! from CG331 HGA2, penalty= 20
ANGLES
HGA3 CG331 CG331 13 100 14 2
HGA3 CG331 HGA3 15 101
DIHEDRALS
HGA2 CG331 CG331 HGA3 1 1 30
HGA3 CG331 CG331 HGA2 2 3 180
HGA1 CG331 CG331 HGA3 4 1 90
NONBONDED
CG331 0 -1 2
HGA3 0 -2 3 0 -4 5
NBFIX
CG331 HGA3 -3 4
END
""",
        encoding="ascii",
    )
    (tmp_path / "par_all36_cgenff_increments.prm").write_text(
        """BONDS
CG331 CG331 10 1 !<% N/A %>!
CG331 HGA3 12 1 !<% .07 %>!
ANGLES
HGA3 CG331 CG331 13 100 !<% 0 0 %>!
HGA3 CG331 HGA3 15 101 !<% N/A %>!
DIHEDRALS
HGA3 CG331 CG331 HGA3 2 3 180 !<% N/A %>!
END
""",
        encoding="ascii",
    )
    ff = V5ForceField.from_directory(tmp_path)
    # The observed bond substitution is inner/bonded; synthetic test adds an
    # independent outer cost for the two-row serialization exercise only.
    return replace(ff, penalties=replace(ff.penalties, nonbonded={("HGA3", "HGA2"): 2}))


def test_stream_multiterm_changed_only_and_named_atoms(bundle, tmp_path):
    mol = Chem.MolFromSmiles("CC")
    mol.GetAtomWithIdx(0).SetProp("_TriposAtomName", "CEt")
    mol.GetAtomWithIdx(1).SetProp("molFileAlias", "H1")  # reserved before generating H names
    result = parameterize_v5(mol, bundle)
    terms = [t for t in result.terms if t.status == "analogy"]
    assert len(terms) == 9 and terms[0].match.record_reversed == (False, True)
    dest = tmp_path / "result.str"
    write_v5_stream(result, dest)
    text = dest.read_text()
    assert "ATOM CEt CG331" in text and result.atoms[2].name == "H2"
    assert "NBFIX" not in text and "MASS" not in text
    assert len(read_topology(dest).residues["LIG"].bonds) == 7
    delta = text.split("read param card flex append")[1]
    assert not any(x in delta for x in ("ANGLES", "BONDS", "NONBONDED"))
    key = "HGA3 CG331 CG331 HGA3"
    assert delta.count(key) == 2
    p = tmp_path / "delta.prm"
    p.write_text(delta, encoding="ascii")
    reloaded = read_parameters(p).match(Kind.PROPER, tuple(key.split()))
    assert [r.values for r in reloaded.records] == [(1, 1, 30), (2, 3, 180)]
    write_v5_stream(result, dest)  # replacing a previous result is allowed
    assert dest.read_text() == text


@pytest.mark.parametrize(
    "case,reason",
    [
        ("name", "duplicate"),
        ("unsafe", "atom name"),
        ("resi", "collides"),
        ("patch", "collides"),
        ("linear", "linear torsions"),
        ("bond", "covalent bond"),
        ("omitted", "AUTO inventory"),
        ("partial", "incomplete"),
        ("collision", "Conflicting analogy"),
        ("net", "sum to residue charge"),
    ],
)
def test_stream_failures_leave_existing_file_unchanged(case, reason, tmp_path, bundle):
    r = parameterize_v5("CC", bundle)
    output = tmp_path / "out.str"
    write_v5_stream(r, output)
    previous = output.read_bytes()
    name = "LIG"
    if case in {"name", "unsafe"}:
        atom = replace(r.atoms[0], name=r.atoms[1].name.lower() if case == "name" else "X\nEND")
        r = replace(r, atoms=(atom, *r.atoms[1:]))
    elif case in {"resi", "patch"}:
        name = "takn" if case == "resi" else "PTCH"
    elif case == "linear":
        r = replace(r, omitted_linear_propers=r.terms[:1])
    elif case == "bond":
        r = replace(r, bonds=r.bonds + r.bonds[:1])
    elif case == "omitted":
        r = replace(r, terms=r.terms[1:])
    elif case == "partial":
        r = replace(r, issues=("unresolved LP",))
    elif case == "net":
        r = replace(r, atoms=(replace(r.atoms[0], formal_charge=3), *r.atoms[1:]))
    else:
        match = bundle.parameters.match(Kind.PROPER, tuple("HGA1 CG331 CG331 HGA3".split()))
        i = next(i for i, t in enumerate(r.terms) if t.status == "analogy")
        r = replace(r, terms=(*r.terms[:i], replace(r.terms[i], match=match), *r.terms[i + 1 :]))
    with pytest.raises(AutoCGENFFError, match=reason):
        write_v5_stream(r, output, residue_name=name)
    assert output.read_bytes() == previous


def test_stream_cli_sdf_alias_audit_and_failure_codes(bundle, tmp_path, capsys):
    # Give the ordinary no-custom-model CLI an exact synthetic proper instead.
    path = tmp_path / "par_all36_cgenff.prm"
    path.write_text(
        path.read_text().replace("DIHEDRALS\n", "DIHEDRALS\nHGA3 CG331 CG331 HGA3 8 1 0\n"),
        encoding="ascii",
    )
    sdf = tmp_path / "one.sdf"
    mol = Chem.MolFromSmiles("CC")
    Chem.SetAtomAlias(mol.GetAtomWithIdx(1), "Z9")
    with Chem.SDWriter(str(sdf)) as writer:
        writer.write(mol)
    out = tmp_path / "one.str"
    args = ["assign-v5", "--force-field", str(tmp_path), "--input", str(sdf), "--stream", str(out)]
    assert main(args) == 0
    audit_complete = json.loads(capsys.readouterr().out)
    assert audit_complete["is_complete"] and audit_complete["stream_written"]["residue"] == "LIG"
    assert audit_complete["atoms"][1]["name"] == "Z9" and "read param" not in out.read_text()
    old = out.read_bytes()
    assert main(args[:3] + ["--smiles", "CCCl", "--stream", str(out)]) == 1
    audit = json.loads(capsys.readouterr().out)
    assert audit["issues"] and not audit["is_complete"] and audit["stream_written"] is None
    assert out.read_bytes() == old
    for unrelated in (b"not our file\n", b"\xff\xfeX\x00"):
        out.write_bytes(unrelated)
        assert main(args) == 2
        channels = capsys.readouterr()
        failed_export = json.loads(channels.out)
        assert failed_export["is_complete"] and failed_export["atoms"] == audit_complete["atoms"]
        assert failed_export["stream_written"] is None
        assert failed_export["stream_error"] == json.loads(channels.err)["error"]
        assert "Refusing to overwrite" in failed_export["stream_error"]
        assert out.read_bytes() == unrelated
