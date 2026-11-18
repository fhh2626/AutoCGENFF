"""Native v5 scientific integration. Only synthetic input records are embedded."""

import os
from itertools import permutations
from math import fsum
from pathlib import Path

import pytest
from rdkit import Chem

from autocgenff import AutoCGENFFError, V5ForceField, parameterize_v5
from autocgenff.charges import assign_charges
from autocgenff.molecule import Kind, Term
from autocgenff.v5 import type_v5
from autocgenff.v5_inputs import read_native_increments

ROOT = Path(__file__).resolve().parents[1]


def test_native_reverse_multiterm_na_missing_and_zero(tmp_path):
    p = tmp_path / "test.prm"
    p.write_text(
        """BONDS
B A 1 2 !<% 0 %>!
B C 1 2 !<% N/A %>!
ANGLES
C B A 3 4 !<% .2 -.3 %>!
DIHEDRALS
D C B A 5 1 0 !<% .01 .02 .03 %>!
D C B A 6 2 0 !<% N/A %>!
IMPROPER
A B C D 7 0 0 !<% 2 2 2 %>!
END
""",
        encoding="utf-8",
    )
    model = read_native_increments(p)
    mol = Chem.MolFromSmiles("CCCC")
    paths = [
        Term(Kind.BOND, (0, 1)),
        Term(Kind.BOND, (1, 2)),
        Term(Kind.ANGLE, (0, 1, 2)),
        Term(Kind.PROPER, (0, 1, 2, 3)),
    ]
    result = assign_charges(mol, tuple("ABCD"), paths, model)
    assert result.charges == pytest.approx((-0.27, 0.49, -0.21, -0.01))
    assert [r.status for r in result.terms] == ["exact", "not_applicable", "exact", "exact"]
    assert [r.reversed for r in result.terms] == [True, False, True, True]
    assert result.terms[-1].source_types == tuple("DCBA") and result.terms[-1].source_line == 7
    assert result.terms[-1].values == (-0.03, -0.02, -0.01)
    missing = assign_charges(mol, tuple("AACC"), paths[:1], model)
    assert missing.charges is None and missing.terms[0].status == "missing"


@pytest.mark.parametrize(
    "row",
    [
        "A B 1 2 !<% NaN %>!",
        "A B 1 2 !<% 0 1 %>!",
        "A A 1 2 !<% .01 %>!",
        "A B 1 2 !<% 0",
        "A B 1 2 !<% 0 %>!\nB A 1 2 !<% .3 %>!",
    ],
)
def test_bad_native_annotations_raise_with_location(tmp_path, row):
    path = tmp_path / "bad.prm"
    path.write_text(f"BONDS\n{row}\nEND\n", encoding="utf-8")
    with pytest.raises(AutoCGENFFError, match=r"bad.prm:\d+:"):
        read_native_increments(path)


def test_sub_and_leaf_actions_separate_from_input_and_partial_charges():
    r = type_v5("CC=[NH+]C")
    assert (
        r.atoms[1].formal_charge == 0 and r.atoms[1].rule_charge == 1 and r.atoms[1].improper_center
    )
    assert r.atoms[2].formal_charge == 1 and r.atoms[2].rule_charge == 0
    ions = type_v5("[Na+].[Cl-].[Ca+2]")
    assert [a.rule_charge for a in ions.atoms] == [1, -1, 2]
    # Equal and opposite unsupported seeds may not cancel between fragments.
    with pytest.raises(AutoCGENFFError, match="preserve formal charge"):
        parameterize_v5("[NH2-].[CH3+]", Path("absent"))


@pytest.mark.parametrize(
    ("smiles", "centers"),
    [
        ("CN=C=O", ()),
        ("CC1(C)C=COCCO1", (4,)),
        ("c1ccco1", ()),  # aromatic five-ring ether is not the seven-ring enol ether
        ("N=c1ccc2ccccc2n1N", (1,)),
        ("Nn1c(-c2ccco2)nc2ccccc2c1=O", (0, 15)),
    ],
)
def test_structure_only_improper_centers(smiles, centers):
    r = type_v5(smiles)
    assert r.is_complete
    assert tuple(a.index for a in r.atoms if a.improper_center) == centers


def test_narrow_train_corrections_do_not_relabel_other_rings_or_open_azines():
    assert type_v5("O=C1CCCC1").types[:2] == ("OG2D3", "CG2O5")
    assert type_v5("O=C1CCCN1").types[:2] == ("OG2D1", "CG2R53")
    assert type_v5("CC(=NN=C(C)C)C").types[1:5] == ("CG2DC1", "NG2D1", "NG2D1", "CG2DC1")


@pytest.fixture(scope="module")
def local():
    directory = Path(os.environ.get("AUTOCGENFF_V5_DIR", ROOT / "force_fields"))
    inc = Path(
        os.environ.get("AUTOCGENFF_V5_INCREMENTS", directory / "par_all36_cgenff_increments.prm")
    )
    if not inc.is_file() or not (directory / "top_all36_cgenff.rtf").is_file():
        pytest.skip("supply local v5 PRM/RTF and AUTOCGENFF_V5_INCREMENTS")
    return V5ForceField.from_directory(directory, increments_path=inc)


# Independent charge oracle: pinned LigandFF IncrementTable + assign_charges on
# named RTF types/bonds (not the prediction), with C2:-1 for ACET, NZ:+1 for
# MAMM, zero elsewhere. These are molecule outputs, NOT mechanical parameters,
# increments or optimized template charges. See docs/local-v5.md.
@pytest.mark.parametrize(
    ("name", "smiles", "order", "expected"),
    [
        (
            "ETOH",
            "CCO",
            "C2 C1 O1 H21 H22 H23 H11 H12 HO1",
            [-0.271, 0.052, -0.651, 0.09, 0.09, 0.09, 0.09, 0.09, 0.42],
        ),
        (
            "MAMM",
            "C[NH3+]",
            "CE NZ HE1 HE2 HE3 HZ1 HZ2 HZ3",
            [0.038, -0.298, 0.09, 0.09, 0.09, 0.33, 0.33, 0.33],
        ),
        (
            "ACET",
            "CC(=O)[O-]",
            "C1 C2 O2 O1 H1 H2 H3",
            [-0.369, 0.619, -0.76, -0.76, 0.09, 0.09, 0.09],
        ),
        (
            "ACEM",
            "CC(=O)N",
            "CC C O N HC1 HC2 HC3 HC HT",
            [-0.275, 0.547, -0.537, -0.625, 0.09, 0.09, 0.09, 0.31, 0.31],
        ),
        (
            "PYR1",
            "c1ccncc1",
            "C4 C3 C2 N1 C6 C5 H4 H3 H2 H6 H5",
            [-0.106, -0.120, 0.174, -0.594, 0.174, -0.120, 0.116, 0.116, 0.122, 0.122, 0.116],
        ),
        ("CO2", "O=C=O", "OC1 C OC2", [-0.342, 0.684, -0.342]),
    ],
)
def test_live_native_molecule_exact_charge_and_inventory(local, name, smiles, order, expected):
    r = parameterize_v5(smiles, local)
    assert r.is_complete and not r.issues and not r.omitted_linear_propers
    assert all(a.mass > 0 for a in r.atoms)
    assert [a.partial_charge for a in r.atoms] == pytest.approx(expected)
    res = local.topology.residues[name]
    assert fsum(a.partial_charge for a in r.atoms) == pytest.approx(res.charge)
    labels = {a.name: a.atom_type for a in res.atoms}
    names = order.split()
    assert [a.atom_type for a in r.atoms] == [labels[n] for n in names]
    bonds = {b.atoms: b.order for b in r.bonds}
    assert set(bonds) == {tuple(sorted((names.index(a), names.index(b)))) for a, b in res.bonds}
    if name == "CO2":
        assert bonds == {(0, 1): 2, (1, 2): 2}
    # Verify every path against named RTF connectivity, not the RDKit generator.
    assert res.auto_angles and res.auto_propers
    for size, kind in [(2, Kind.BOND), (3, Kind.ANGLE), (4, Kind.PROPER)]:
        expected_paths = {
            p
            for p in permutations(range(len(names)), size)
            if p < p[::-1]
            and all(tuple(sorted((names[i], names[j]))) in res.bonds for i, j in zip(p, p[1:]))
        }
        paths = [t.term.atoms for t in r.terms if t.term.kind == kind]
        assert len(paths) == len(expected_paths) and set(paths) == expected_paths
    impropers = [t for t in r.terms if t.term.kind == Kind.IMPROPER]
    assert [t.target_types for t in impropers] == [
        tuple(labels[n] for n in p) for p in res.impropers
    ]
    assert all(t.penalty == 0 and t.source_line is not None for t in r.charge_diagnostics.terms)
    if name == "ACET":
        assert [a.rule_charge for a in r.atoms[:4]] == [0, -1, 0, 0]
        assert [a.formal_charge for a in r.atoms[:4]] == [0, 0, 0, -1]


@pytest.mark.parametrize(
    ("name", "smiles", "order"),
    [
        ("MICY", "CN=C=O", "C1 N C O H11 H12 H13"),
        (
            "2300N9",
            "CC1(C)C=COCCO1",
            "C01 C2 C1 C9 C10 O11 C12 C13 O14 H01 H02 H03 H15 H16 H17 H23 H24 H25 H26 H27 H28",
        ),
        (
            "23000S",
            "N=c1ccc2ccccc2n1N",
            "N1 C2 C3 C4 C5 C6 C7 C8 C9 C10 N11 N12 H13 H14 H15 H16 H17 H18 H19 H20 H21",
        ),
        (
            "23000F",
            "Nn1c(-c2ccco2)nc2ccccc2c1=O",
            "N1 N2 C12 C13 C14 C15 C16 O17 N11 C10 C9 C8 C7 C6 C5 C3 O4 "
            "H19 H18 H24 H25 H26 H23 H22 H21 H20",
        ),
        (
            "2300BC",
            "NCc1nc2ccccc2c(=O)n1N",
            "N14 C13 C12 N11 C10 C9 C8 C7 C6 C5 C3 O4 N2 N1 "
            "H01 H02 H24 H25 H23 H22 H21 H20 H19 H18",
        ),
        (
            "2300BB",
            "NN1C(=O)c2ccccc2N=C1C[NH3+]",
            "N1 N2 C3 O4 C5 C6 C7 C8 C9 C10 N11 C12 C13 N14 "
            "H19 H18 H20 H21 H22 H23 H24 H25 H15 H16 H17",
        ),
    ],
)
def test_live_improper_inventory_matches_named_reference(local, name, smiles, order):
    r = parameterize_v5(smiles, local)
    ref = local.topology.residues[name]
    labels = {a.name: a.atom_type for a in ref.atoms}
    names = order.split()
    assert r.is_complete, r.issues
    assert set(names) == labels.keys()
    assert [a.atom_type for a in r.atoms] == [labels[n] for n in names]
    bonds = {tuple(sorted(names[i] for i in b.atoms)) for b in r.bonds}
    assert bonds == set(ref.bonds)
    impropers = [t for t in r.terms if t.term.kind == Kind.IMPROPER]
    assert sorted(tuple(names[i] for i in t.term.atoms) for t in impropers) == sorted(ref.impropers)
    assert all(t.status == "exact" and t.match.records[0].values[0] != 0 for t in impropers)
    assert {names[a.index] for a in r.atoms if a.improper_center} == {q[0] for q in ref.impropers}


def test_live_neopentyl_alcohol_transfers_by_observed_analogy(local):
    r = parameterize_v5("CC(C)(C)CO", local)
    assert r.is_complete and fsum(a.partial_charge for a in r.atoms) == pytest.approx(0)
    bonded = [t for t in r.terms if t.status == "analogy"]
    inc = [t for t in r.charge_diagnostics.terms if t.status == "analogy"]
    assert len(bonded) == len(inc) == 3
    assert "provisional" in r.substitution_profile
    for mechanical, charge in zip(bonded, inc, strict=True):
        assert mechanical.term == charge.term and mechanical.term.kind == Kind.PROPER
        assert mechanical.target_types == ("CG331", "CG301", "CG321", "OG311")
        assert (
            mechanical.match.records[0].types
            == charge.source_types
            == ("CG321", "CG301", "CG321", "OG311")
        )
        assert charge.source_line and mechanical.penalty == charge.penalty
        assert any(
            (e.target, e.source, e.matrix, e.cost)
            == ("CG331", "CG321", "nonbonded", charge.penalty)
            for e in local.substitution_evidence
        )


def test_live_linear_omissions_do_not_hide_missing_increments_or_lone_pairs(local):
    r = parameterize_v5("CC#N", local)
    assert not r.is_complete and len(r.omitted_linear_propers) == 3
    assert not any(t.term.kind == Kind.PROPER for t in r.terms)
    assert {t.term for t in r.omitted_linear_propers} == {
        t.term for t in r.charge_diagnostics.terms if t.status == "missing"
    }
    for smiles in ("c1ccccc1I", "CCCl"):
        r = parameterize_v5(smiles, local)
        assert not r.is_complete and any("lone-pair" in issue for issue in r.issues)
        assert all(a.partial_charge is None for a in r.atoms)


@pytest.mark.parametrize("use_exact", [True, False])
def test_linear_center_exact_multiterm_precedes_omission_and_not_wildcard(tmp_path, use_exact):
    # Synthetic mechanical records only; all other dependencies are unresolved.
    key = "HGA3 CG331 CG1N1 NG1T1"
    prm = "DIHEDRALS\nX CG331 CG1N1 X 2 1 0\n"
    if use_exact:
        prm += f"{key} 3 1 0\n{key} 4 2 180\n"
    (tmp_path / "par_all36_cgenff.prm").write_text(prm + "END\n", encoding="utf-8")
    (tmp_path / "top_all36_cgenff.rtf").write_text("RESI SYN 0\nEND\n", encoding="utf-8")
    (tmp_path / "par_all36_cgenff_increments.prm").write_text(
        "BONDS\nHGA3 CG331 3 2 !<% N/A %>!\nEND\n", encoding="utf-8"
    )
    r = parameterize_v5("CC#N", tmp_path)
    assert not r.is_complete
    assert len(r.omitted_linear_propers) == (0 if use_exact else 3)
    propers = [t for t in r.terms if t.term.kind == Kind.PROPER]
    assert len(propers) == (3 if use_exact else 0)
    assert all(t.status == "exact" and len(t.match.records) == 2 for t in propers)


def test_improper_ambiguous_orders_never_choose_by_atom_index(tmp_path):
    (tmp_path / "par_all36_cgenff.prm").write_text(
        "IMPROPER\nCG2O1 CG331 NG2S2 OG2D1 2 0 0\nCG2O1 NG2S2 CG331 OG2D1 3 0 0\nEND\n",
        encoding="utf-8",
    )
    (tmp_path / "top_all36_cgenff.rtf").write_text("RESI SYN 0\nEND\n", encoding="utf-8")
    (tmp_path / "par_all36_cgenff_increments.prm").write_text(
        "BONDS\nHGA3 CG331 3 2 !<% 0 %>!\nEND\n", encoding="utf-8"
    )
    r = parameterize_v5("CC(=O)N", tmp_path)
    assert not r.is_complete
    (impr,) = [t for t in r.terms if t.term.kind == Kind.IMPROPER]
    assert impr.status == "ambiguous_improper_order" and impr.match is None
