"""Native v5 scientific integration. Only synthetic input records are embedded."""

import os
from itertools import permutations
from math import fsum
from pathlib import Path

import pytest
from rdkit import Chem

from autocgenff import (
    AutoCGENFFError,
    V5ForceField,
    bundled_force_field_dir,
    parameterize_v5,
    write_v5_stream,
)
from autocgenff.charges import assign_charges
from autocgenff.molecule import Kind, Term
from autocgenff.v5 import type_v5
from autocgenff.v5_inputs import read_native_increments


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
        "A B C 2 !<% .1 %>!",
        "A B 1 2 3 !<% .1 %>!",
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
    ("bridged", "alternate", "carbon_type"),
    [
        ("CC(N)=[N+](C)C=N", "CC(=[NH2+])N(C)C=N", "CG2N2"),
        ("NC(N)=[N+](C)C(=N)N", "NC(=[NH2+])N(C)C(=N)N", "CG2N1"),
    ],
)
def test_amidinium_seed_requires_the_double_bonded_nitrogen_to_be_positive(
    bridged: str, alternate: str, carbon_type: str
) -> None:
    for smiles in (bridged, alternate):
        result = type_v5(smiles)
        assert result.is_complete
        assert result.types[1] == result.types[5] == carbon_type
        assert [(a.index, a.rule_charge) for a in result.atoms if a.rule_charge] == [(1, 1)]


def test_distinct_positive_nitrogens_keep_both_carbon_seeds() -> None:
    result = type_v5("C(=[NH2+])NC(=[NH2+])N")
    assert result.is_complete
    assert result.types[0] == "CG2N2" and result.types[3] == "CG2N1"
    assert [(a.index, a.rule_charge) for a in result.atoms if a.rule_charge] == [(0, 1), (3, 1)]


@pytest.mark.parametrize("smiles", ["C=[N+]=C", "CC(N)=[N+]=C"])
def test_cumulene_seed_overlap_still_refuses_assignment(smiles: str) -> None:
    with pytest.raises(AutoCGENFFError, match="preserve formal charge"):
        parameterize_v5(smiles, Path("absent"))


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
    env_dir = os.environ.get("AUTOCGENFF_V5_DIR")
    directory = Path(env_dir) if env_dir is not None else bundled_force_field_dir()
    env_inc = os.environ.get("AUTOCGENFF_V5_INCREMENTS")
    inc = Path(env_inc) if env_inc is not None else directory / "par_all36_cgenff_increments.prm"
    if (
        not inc.is_file()
        or not (directory / "par_all36_cgenff.prm").is_file()
        or not (directory / "top_all36_cgenff.rtf").is_file()
    ):
        pytest.skip("supply local v5 PRM/RTF and AUTOCGENFF_V5_INCREMENTS")
    return V5ForceField.from_directory(directory, increments_path=inc)


# Reference charge totals on named RTF types/bonds (not the prediction), with
# C2:-1 for ACET, NZ:+1 for MAMM, zero elsewhere. These are molecule outputs,
# NOT mechanical parameters, increments or optimized template charges. See
# docs/local-v5.md.
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


def test_live_linear_center_paths_and_halogen_lone_pairs(local, tmp_path):
    assert local.charges.linear_centers == {"CG1N1", "CG1T1", "CG1T2"}
    # The carbon-bound azide N is also NG1T1 but bent; its torsions stay ordinary paths.
    azide = parameterize_v5("CN=[N+]=[N-]", local)
    assert all(
        t.status != "linear_not_applicable" for t in azide.charge_diagnostics.terms
    )
    r = parameterize_v5("CC#N", local)
    assert r.is_complete and len(r.omitted_linear_propers) == 3
    assert not any(t.term.kind == Kind.PROPER for t in r.terms)
    assert {t.term for t in r.omitted_linear_propers} == {
        t.term for t in r.charge_diagnostics.terms if t.status == "linear_not_applicable"
    }
    assert all(t.status != "missing" for t in r.charge_diagnostics.terms)
    # Same values as the published ACN residue.
    assert [a.partial_charge for a in r.atoms] == pytest.approx(
        (-0.17, 0.36, -0.46, 0.09, 0.09, 0.09)
    )
    for smiles, expected_dist, expected_host_q in (
        ("Clc1ccccc1", 1.64, -0.20),
        ("Brc1ccccc1", 1.89, -0.181),
        ("Ic1ccccc1", 2.20, -0.17),
    ):
        hal = parameterize_v5(smiles, local)
        assert hal.is_complete and len(hal.lone_pairs) == 1
        lp = hal.lone_pairs[0]
        assert lp.atom_type == "LPH" and lp.distance == pytest.approx(expected_dist)
        assert lp.charge == pytest.approx(0.05) and lp.scale == 0.0
        assert hal.atoms[lp.host].partial_charge == pytest.approx(expected_host_q, abs=1e-3)
        total_q = sum(a.partial_charge for a in hal.atoms) + sum(s.charge for s in hal.lone_pairs)
        assert total_q == pytest.approx(0.0, abs=1e-9)
        stream = tmp_path / f"hal_{lp.host}_{int(expected_dist * 100)}.str"
        write_v5_stream(hal, stream, residue_name="HLIG")
        text = stream.read_text(encoding="ascii")
        assert f"ATOM {lp.name} LPH 0.05" in text
        assert f"LONEPAIR COLINEAR {lp.name} " in text and "SCALE 0.0" in text
    aliphatic = parameterize_v5("CCCl", local)
    assert aliphatic.is_complete and not aliphatic.lone_pairs
    cl_atom = next(a for a in aliphatic.atoms if a.element == "Cl")
    assert cl_atom.partial_charge == pytest.approx(-0.10, abs=1e-6)


def test_halogen_lone_pair_fails_closed_without_lph_in_base_ff(tmp_path):
    (tmp_path / "par_all36_cgenff.prm").write_text("BONDS\nCG2R61 CLGR1 300 1.7\nEND\n", encoding="utf-8")
    (tmp_path / "top_all36_cgenff.rtf").write_text("RESI SYN 0\nEND\n", encoding="utf-8")
    (tmp_path / "par_all36_cgenff_increments.prm").write_text(
        "BONDS\nCG2R61 CLGR1 300 1.7 !<% 0 %>!\nEND\n", encoding="utf-8"
    )
    r = parameterize_v5("Clc1ccccc1", tmp_path)
    assert not r.is_complete and not r.lone_pairs
    assert any("LPH" in issue for issue in r.issues)
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


@pytest.mark.parametrize(
    ("annotation", "status", "expected"),
    [
        (None, "missing", None),
        ("N/A", "not_applicable", (0, 0, 0, 0, 0, 0)),
        (".01 .02 .03", "exact", (-.03, -.03, .09, -.01, -.01, -.01)),
    ],
)
def test_linear_mechanical_omission_does_not_decide_charge_transfer(
    tmp_path: Path,
    annotation: str | None,
    status: str,
    expected: tuple[float, ...] | None,
) -> None:
    prm = """BONDS
HGA3 CG331 1 2 !<% N/A %>!
CG331 CG1N1 1 2 !<% N/A %>!
CG1N1 NG1T1 1 2 !<% N/A %>!
ANGLES
HGA3 CG331 HGA3 3 4 !<% N/A %>!
HGA3 CG331 CG1N1 3 4 !<% N/A %>!
CG331 CG1N1 NG1T1 3 4 !<% N/A %>!
"""
    increments = prm
    if annotation is not None:
        increments += f"DIHEDRALS\nHGA3 CG331 CG1N1 NG1T1 5 1 0 !<% {annotation} %>!\n"
    (tmp_path / "par_all36_cgenff.prm").write_text(prm + "END\n", encoding="utf-8")
    (tmp_path / "top_all36_cgenff.rtf").write_text("RESI SYN 0\nEND\n", encoding="utf-8")
    (tmp_path / "par_all36_cgenff_increments.prm").write_text(increments + "END\n", encoding="utf-8")
    result = parameterize_v5("CC#N", tmp_path)
    omitted = {item.term for item in result.omitted_linear_propers}
    assert len(omitted) == 3
    proper_charges = [
        item for item in result.charge_diagnostics.terms if item.term.kind == Kind.PROPER
    ]
    assert {item.term for item in proper_charges} == omitted
    assert all(item.status == status for item in proper_charges)
    if expected is None:
        assert result.charge_diagnostics.charges is None
    else:
        assert result.charge_diagnostics.charges == pytest.approx(expected)


@pytest.mark.parametrize(
    ("evidence", "linear_status"),
    [
        (None, "missing"),
        ("N/A", "linear_not_applicable"),
        (".01 .02 .03", "missing"),  # one numeric sp-center row disables the rule
        ("N/A unannotated", "missing"),  # so does a row without an increment field
    ],
)
def test_unlisted_sp_center_paths_follow_native_na_rows_only(
    tmp_path: Path, evidence: str | None, linear_status: str
) -> None:
    increments = "BONDS\nHGA3 CG331 1 2 !<% N/A %>!\n"
    if evidence is not None:
        # Other sp-center torsion rows; CCC#N itself has no proper rows.
        increments += "DIHEDRALS\nCG2R61 CG2R61 CG1N1 NG1T1 5 1 0 !<% "
        increments += evidence.removesuffix(" unannotated") + " %>!\n"
        if evidence.endswith("unannotated"):
            increments += "CG321 SG311 CG1N1 NG1T1 5 1 0\n"
    (tmp_path / "par_all36_cgenff.prm").write_text(
        "BONDS\nHGA3 CG331 322 1.111\nEND\n", encoding="utf-8"
    )
    (tmp_path / "top_all36_cgenff.rtf").write_text("RESI SYN 0\nEND\n", encoding="utf-8")
    (tmp_path / "par_all36_cgenff_increments.prm").write_text(increments + "END\n", encoding="utf-8")
    result = parameterize_v5("CCC#N", tmp_path)
    statuses = {}
    for item in result.charge_diagnostics.terms:
        if item.term.kind == Kind.PROPER:
            center = set(item.target_types[1:3])
            statuses.setdefault("CG1N1" in center, set()).add(item.status)
    # A terminal sp atom does not make the torsion undefined.
    assert statuses == {True: {linear_status}, False: {"missing"}}
    assert result.charge_diagnostics.charges is None


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


@pytest.mark.parametrize(("potential", "status"), [("40 0 0", "exact"), ("40 2 90", "ambiguous_improper_order")])
def test_improper_jk_swap_requires_an_even_potential(tmp_path, potential, status):
    (tmp_path / "par_all36_cgenff.prm").write_text(
        f"IMPROPER\nCG2O1 CG331 NG2S2 OG2D1 {potential}\n"
        f"CG2O1 NG2S2 CG331 OG2D1 {potential}\nEND\n",
        encoding="utf-8",
    )
    (tmp_path / "top_all36_cgenff.rtf").write_text("RESI SYN 0\nEND\n", encoding="utf-8")
    (tmp_path / "par_all36_cgenff_increments.prm").write_text(
        "BONDS\nHGA3 CG331 3 2 !<% 0 %>!\nEND\n", encoding="utf-8"
    )
    r = parameterize_v5("CC(=O)N", tmp_path)
    (impr,) = [t for t in r.terms if t.term.kind == Kind.IMPROPER]
    assert impr.status == status
    assert (impr.match is not None) == (status == "exact")
    if impr.match is not None:
        assert impr.target_types == ("CG2O1", "CG331", "NG2S2", "OG2D1")


def test_live_native_improper_orders_resolve_only_without_conflicts(local):
    for smiles in (
        "NC(=[NH2+])NCCC[C@H]([NH3+])C(=O)[O-]",  # arginine
        "[NH3+][C@@H](Cc1c[nH+]c[nH]1)C(=O)[O-]",  # protonated histidine
    ):
        r = parameterize_v5(smiles, local)
        assert r.is_complete, f"{smiles}: {r.issues}"
    adenine = parameterize_v5("Nc1ncnc2[nH]cnc12", local)
    assert any(
        t.status == "ambiguous_improper_order" and t.target_types[0] == "NG2S3"
        for t in adenine.terms
    )


def test_candidate_index_matches_exhaustive_best_analogy(local):
    from autocgenff.analogy import CandidateIndex, best_analogy

    for kind, target in (
        (Kind.BOND, ("CG2R61", "NG3C51")),
        (Kind.ANGLE, ("HGR61", "CG2R61", "NG3C51")),
        (Kind.PROPER, ("CG331", "CG301", "CG321", "OG311")),
    ):
        candidates = [group[0].types for group in local.parameters.iter_groups(kind)]
        idx = CandidateIndex(kind, candidates)
        plain = best_analogy(kind, target, candidates, local.penalties)
        indexed = best_analogy(kind, target, (), local.penalties, candidate_index=idx)
        assert indexed == plain and indexed is not None


def test_topology_parses_colinear_lonepair_and_rejects_other_modes(tmp_path):
    from autocgenff.topology import read_topology

    rtf = tmp_path / "lp.rtf"
    rtf.write_text(
        "RESI GOOD 0\n"
        "ATOM CL1 CLGR1 -0.2\n"
        "ATOM C1 CG2R61 0.15\n"
        "ATOM LP1 LPH 0.05\n"
        "BOND CL1 C1\n"
        "LONEPAIR COLINEAR LP1 CL1 C1 DIST 1.64 SCALE 0.0\n"
        "RESI BAD 0\n"
        "ATOM O1 OG311 -0.5\n"
        "ATOM LP1 LPH 0.5\n"
        "LONEPAIR BISE LP1 O1 O1 O1 DIST 0.5\n"
        "RESI DUP 0\n"
        "ATOM CL1 CLGR1 -0.2\n"
        "ATOM C1 CG2R61 0.15\n"
        "ATOM LP1 LPH 0.05\n"
        "LONEPAIR COLINEAR LP1 CL1 C1 DIST 1.64 DIST 1.70\n"
        "END\n",
        encoding="utf-8",
    )
    topo = read_topology(rtf)
    good = topo.residues["GOOD"]
    assert not good.unsupported and len(good.lone_pairs) == 1
    assert good.lone_pairs[0].name == "LP1" and good.lone_pairs[0].distance == pytest.approx(1.64)
    bad = topo.residues["BAD"]
    assert "LONEPAIR" in bad.unsupported and not bad.lone_pairs
    dup = topo.residues["DUP"]
    assert "LONEPAIR" in dup.unsupported and not dup.lone_pairs


def test_improper_index_cache_never_aliases_distinct_center_maps(tmp_path):
    from autocgenff.analogy import PenaltyModel
    from autocgenff.assignment import _lookup_term
    from autocgenff.parameters import read_parameters

    prm = tmp_path / "impr.prm"
    prm.write_text("IMPROPER\nA B C D 10 0 0\nEND\n", encoding="utf-8")
    params = read_parameters(prm)
    model = PenaltyModel(bonded={("E", "A"): 1.0})
    status0, _, match0 = _lookup_term(
        params, Kind.IMPROPER, ("E", "B", "C", "D"), model, 0, {("A", "B", "C", "D"): 0}
    )
    assert status0 == "analogy" and match0 is not None
    # Check a reused map mutated in place as well as freshly allocated maps.
    centers = {("A", "B", "C", "D"): 0}
    target = ("E", "B", "C", "D")
    assert _lookup_term(params, Kind.IMPROPER, target, model, 0, centers)[0] == "analogy"
    centers[("A", "B", "C", "D")] = 3
    for mapping in (centers, dict(centers)):
        assert _lookup_term(params, Kind.IMPROPER, target, model, 0, mapping) == (
            "missing", None, None
        )


def test_candidate_index_validates_kind_and_skips_wrong_arity():
    from autocgenff.analogy import CandidateIndex, PenaltyModel, best_analogy

    idx = CandidateIndex(Kind.BOND, [("A", "B"), ("BAD",)])
    model = PenaltyModel(bonded={("C", "A"): 1.0})
    assert best_analogy(Kind.BOND, ("C", "B"), (), model, candidate_index=idx) is not None
    with pytest.raises(AutoCGENFFError, match="kind"):
        best_analogy(Kind.ANGLE, ("C", "B", "D"), (), model, candidate_index=idx)
    for index in (idx, None):
        with pytest.raises(AutoCGENFFError, match="Invalid type tuple"):
            best_analogy(Kind.BOND, ("C",), [("A",)], model, candidate_index=index)


def test_penalty_fit_uses_later_row_and_combines_repeated_substitutions(tmp_path):
    from autocgenff.parameters import read_parameters
    from autocgenff.v5_inputs import observed_v5_penalty_model

    prm = tmp_path / "later.prm"
    prm.write_text(
        "DIHEDRALS\nA C D A 1 1 0\nB C D B 2 1 0\n"
        "B C D B 2 2 0 ! from A C D A, penalty=6.0\nEND\n",
        encoding="utf-8",
    )
    model, evidence = observed_v5_penalty_model(read_parameters(prm))
    assert model.nonbonded[("B", "A")] == 3
    assert [(e.line, e.cost) for e in evidence] == [(4, 3)]


def test_penalty_fit_aligns_center_last_improper_annotations(tmp_path):
    from autocgenff.parameters import read_parameters
    from autocgenff.topology import read_topology
    from autocgenff.v5_inputs import observed_v5_penalty_model

    rtf = tmp_path / "centers.rtf"
    rtf.write_text(
        "RESI RESA 0\nATOM W Q 0\nATOM X A 0\nATOM Y B 0\nATOM Z C 0\n"
        "BOND W X W Y W Z\nIMPR W X Y Z\n"
        "RESI RESB 0\nATOM W R 0\nATOM X A 0\nATOM Y B 0\nATOM Z C 0\n"
        "BOND W X W Y W Z\nIMPR Z Y X W\nEND\n",
        encoding="utf-8",
    )
    prm = tmp_path / "centers.prm"
    prm.write_text(
        "IMPROPER\nQ A B C 5 0 0\nC B A R 5 0 0 ! from Q A B C, penalty=20.0\nEND\n",
        encoding="utf-8",
    )
    model, _ = observed_v5_penalty_model(read_parameters(prm), read_topology(rtf))
    assert model.bonded[("R", "Q")] == 2
    _, unverified = observed_v5_penalty_model(read_parameters(prm))
    assert not unverified


def test_custom_precision_lone_pair_round_trips_stream(local, tmp_path):
    from dataclasses import replace

    hal = parameterize_v5("Clc1ccccc1", local)
    lp = replace(hal.lone_pairs[0], distance=1.64005, scale=0.05)
    custom = replace(hal, lone_pairs=(lp,))
    stream = tmp_path / "precise_lp.str"
    write_v5_stream(custom, stream, residue_name="PLIG")
    text = stream.read_text(encoding="ascii")
    assert "DIST 1.64005 SCALE 0.05" in text

    invalid = replace(hal, lone_pairs=(replace(lp, host=float(lp.host)),))
    with pytest.raises(AutoCGENFFError, match="Invalid lone-pair indices"):
        write_v5_stream(invalid, stream, residue_name="PLIG")
    assert stream.read_text(encoding="ascii") == text


def test_observed_alkyl_model_and_native_increments_header_edges(tmp_path):
    from autocgenff.parameters import read_parameters
    from autocgenff.v5_inputs import observed_alkyl_model, read_native_increments

    prm = tmp_path / "alkyl_later.prm"
    prm.write_text(
        "DIHEDRALS\n"
        "CG321 CG321 CG321 CG321 1 1 0\n"
        "CG331 CG321 CG321 CG321 2 1 0\n"
        "CG331 CG321 CG321 CG321 2 2 0 ! from CG321 CG321 CG321 CG321, penalty=4.0\n"
        "END\n",
        encoding="utf-8",
    )
    model, evidence = observed_alkyl_model(read_parameters(prm))
    assert model.nonbonded[("CG331", "CG321")] == 4.0
    assert [(e.line, e.cost) for e in evidence] == [(4, 4.0)]

    inc = tmp_path / "inc.prm"
    inc.write_text(
        "BONDS\n"
        "CG331 CG321 10 1.5 !<% 0.05 %>!\n"
        "MASS\n"
        "CG1N1 CG1T1 CG1T2 CG331 1 1 0\n"
        "END\n",
        encoding="utf-8",
    )
    loaded = read_native_increments(inc)
    assert loaded.linear_centers == frozenset()


def test_check_residue_rejects_unknown_improper_atom_name_as_autocgenff_error(local):
    from autocgenff.topology import Residue, TemplateAtom
    from autocgenff.v5_stream import _check_residue

    r = parameterize_v5("CC(=O)N", local)
    bad = Residue(
        name="LIG",
        charge=0.0,
        line=1,
        auto_angles=True,
        auto_propers=True,
        atoms=[TemplateAtom(a.name.upper(), a.atom_type, a.partial_charge or 0.0) for a in r.atoms],
        bonds={tuple(sorted((r.atoms[b.atoms[0]].name.upper(), r.atoms[b.atoms[1]].name.upper()))): b.order for b in r.bonds},
        impropers=[("UNKNOWN", r.atoms[0].name.upper(), r.atoms[2].name.upper(), r.atoms[3].name.upper())],
    )
    with pytest.raises(AutoCGENFFError, match="unknown or lone-pair atom"):
        _check_residue(bad, r)
