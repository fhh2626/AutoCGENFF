"""Structure-only local-v5 regressions. The typer never receives a reference type or residue."""

import json

import pytest
from rdkit import Chem

from autocgenff.cli import main
from autocgenff.errors import AutoCGENFFError
from autocgenff.v5 import _aromaticity, type_v5, v5_vocabulary
from autocgenff.v5._rules import RuleSyntaxError, parse


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ("rne ring (el C) (el C) (el N)", "at most two ring-neighbor groups"),
        ("iminium_charge charge 0", "only one charge action"),
        ("charge 0 iminium_charge", "only one charge action"),
    ],
)
def test_rule_parser_rejects_impossible_or_conflicting_rules(condition: str, message: str) -> None:
    with pytest.raises(RuleSyntaxError, match=message):
        parse(f"cat main\ntyp TEST : {condition}\nend\n")


@pytest.mark.parametrize(
    ("smiles", "labels"),
    [
        ("[Na+].[Cl-].[Ca+2]", "SOD CLA CAL"),
        ("CO", "CG331 OG311 HGA3 HGA3 HGA3 HGP1"),
        ("F[Al-](F)(F)F", "FGP1 ALG1 FGP1 FGP1 FGP1"),
        ("CB(O)[O-]", "CG331 BG201 OG311 OG2D2 HGA3 HGA3 HGA3 HGP1"),
        ("C[NH3+]", "CG334 NG3P3 HGA3 HGA3 HGA3 HGP2 HGP2 HGP2"),
        ("O=C=O", "OG2D5 CG2O7 OG2D5"),
        ("O=C1OCC1", "OG2D1 CG2O4 OG3C51 CG3C41 CG3C41 HGA2 HGA2 HGA2 HGA2"),
        ("C=CC=C", "CG2DC3 CG2DC1 CG2DC2 CG2DC3 HGA5 HGA5 HGA4 HGA4 HGA5 HGA5"),
        ("O=P([O-])([O-])O", "OG2P1 PG2 OG2P1 OG2P1 OG311 HGP1"),
        ("c1ccco1", "CG2R51 CG2R51 CG2R51 CG2R51 OG2R50 HGR52 HGR51 HGR51 HGR52"),
        ("N#CC#C", "NG1T1 CG1N1 CG1T1 CG1T2 HGPAM1"),
        ("CC(=O)N", "CG331 CG2O1 OG2D1 NG2S2 HGA3 HGA3 HGA3 HGP1 HGP1"),
    ],
)
def test_structure_only_all_atoms(smiles, labels):
    result = type_v5(smiles)
    assert result.is_complete and result.types == tuple(labels.split())
    assert all(a.index == i and a.trace[-1].split()[1] == "typ" for i, a in enumerate(result.atoms))


@pytest.mark.parametrize(
    ("smiles", "unknown"),
    [
        ("[OH-]", (0, 1)),
        ("[OH3+]", (0, 1, 2, 3)),
        ("[H-]O[H]", (0, 1, 2)),
        ("[F-]", (0,)),
        ("[Br-]", (0,)),
        ("[I-]", (0,)),
        ("[K+]", (0,)),
        ("C[Na]", (1,)),
        ("[Na+].[I-]", (1,)),
        ("CI", (1,)),
        ("ClCl", (0, 1)),
        ("FN(C)C", (0,)),
        ("F[N+](C)(C)C", (0,)),
        ("C[Se]C", (1,)),
        ("B(O)(O)O", (0,)),
        ("[B-](F)(F)(F)F", (0, 1, 2, 3, 4)),
        ("FC(F)(F)F", (0, 1, 2, 3, 4)),
        ("F[Al](F)F", (0, 1, 2, 3)),
        ("Cl[Al-](F)(F)F", (0, 1, 2, 3, 4)),
    ],
)
def test_no_guess_outside_supported_environments(smiles, unknown):
    result = type_v5(smiles)
    assert not result.is_complete
    assert tuple(a.index for a in result.atoms if a.atom_type is None) == unknown
    assert all(result.atoms[i].errors for i in unknown)
    assert not ({"HGTIP3", "OGTIP3"} & set(result.types))


@pytest.mark.parametrize("smiles", ["O", "[2H][18O][3H]", "C.O", "[2H]O.CCO"])
def test_water_delegated(smiles):
    with pytest.raises(AutoCGENFFError, match=r"toppar_water_ions\.str"):
        type_v5(smiles)


@pytest.mark.parametrize("smiles", ["[CH3]", "[Cl]", "[OH]", "[Ca+]"])
def test_radicals_reject_whole_input(smiles):
    with pytest.raises(AutoCGENFFError, match="radical"):
        type_v5(smiles)


def test_perception_does_not_change_input_and_ignores_metadata():
    mol = Chem.MolFromSmiles("c1ccncc1")
    mol.SetProp("_Name", "CO2")
    mol.GetAtomWithIdx(0).SetProp("atom_type", "SOD")
    result = type_v5(mol)
    assert result.types == type_v5("C1=CC=NC=C1").types
    assert result.types[0] == "CG2R61"
    assert mol.GetNumAtoms() == 6 and all(b.GetIsAromatic() for b in mol.GetBonds())


def test_type_queries_are_on_demand_including_explicit_h_first():
    result = type_v5("[H]N(C)C")
    assert result.types[:4] == ("HGPAM1", "NG311", "CG3AM1", "CG3AM1")
    assert result.types[4:] == ("HGAAM1",) * 6
    assert "HGN:" in result.atoms[0].trace[-1]


def test_ring_membership_fused_spiro_and_large_rings():
    assert type_v5("C1CC2CCC12").types[2] == "CG3RC1"
    assert type_v5("C1CCC2(CC1)CCCC2").types[3] == "CG3C50"
    assert type_v5("C1CCCCCCC1").types[:8] == ("CG321",) * 8
    assert type_v5("n1ccccc1-c1ccccn1").types[5:7] == ("CG2R67", "CG2R67")


def test_perception_nonconvergence_is_fatal(monkeypatch):
    monkeypatch.setattr(_aromaticity, "MAX_ITERATIONS", 1)
    with pytest.raises(AutoCGENFFError, match="did not converge"):
        type_v5("c1ccccc1")


def test_vocabulary_alternates_are_concrete_not_virtual():
    labels = v5_vocabulary()
    assert len(labels) == 163 and not {"HGTIP3", "OGTIP3"} & labels
    assert {
        "CG2D1O",
        "CG2D2O",
        "CG251O",
        "CG252O",
        "CG25C1",
        "CG25C2",
        "CG2DC1",
        "CG2DC2",
    } <= labels
    assert not {"LPH", "UNK"} & labels and not any("?" in label for label in labels)


def test_cli_type_only_and_separate_from_assignment(capsys):
    assert main(["type-v5", "--smiles", "CO"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["profile"] == "local-v5" and report["is_complete"]
    assert report["atoms"][0]["formal_charge"] == 0
    assert not {"charges", "impropers", "terms"} & report.keys()
    assert main(["type-v5", "--smiles", "[Br-]"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["is_complete"] and report["atoms"][0]["atom_type"] is None
    for args in (
        ["type-v5", "--smiles", "O"],
        ["--smiles", "O"],
        ["type-v5", "--smiles", "O", "--rules", "override.json"],
        ["type-v5", "--smiles", "[Cl]"],
    ):
        assert main(args) == 2
        assert "error" in json.loads(capsys.readouterr().err)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("O=NCC", {0: "OG2D1", 1: "NG2D1"}),
        ("C[N+](=O)[O-]", {1: "NG2O1", 2: "OG2N1", 3: "OG2N1"}),
        ("C[N+](C)(C)[O-]", {4: "OG312"}),
        ("O=C1NS(=O)(=O)C=C1", {1: "CG2R53", 2: "NG2R53"}),
        ("O=C1N(C)C=CC1=O", {2: "NG2R51"}),
        ("CC=C1CCCC1", {1: "CG2D1", 2: "CG2D1"}),
        ("C=CC=C1CCCC1", {2: "CG2DC2", 3: "CG25C2"}),
        ("C=C1CCCN1", {1: "CG251O"}),
    ],
)
def test_selected_motif_typing(smiles: str, expected: dict[int, str]) -> None:
    result = type_v5(smiles)
    assert result.is_complete
    assert {i: result.types[i] for i in expected} == expected


@pytest.mark.parametrize(
    ("smiles", "index", "expected"),
    [
        ("CC1(C)CCCC(C)(C)[N+]1=O", 10, "OG2N1"),
        ("C#CC=C1CCCC1", 3, "CG25C1"),
        ("COC=C1CCCC1", 3, "CG25C1"),
        ("c1ccccc1C=C2CCCC2", 7, "CG25C1"),
        ("O=C1C=CC2N1S(=O)(=O)CC=2", 5, "NG2R51"),
        ("O=C1C=CC2N1S(=O)(=O)C(=O)C=2", 5, "NG2R51"),
        ("O=C1[N-]S(=O)(=O)C=C1", 2, "NG2R51"),
        ("S=C1NS(=O)(=O)C=C1", 2, "NG2R51"),
    ],
)
def test_neighboring_motif_typing(smiles: str, index: int, expected: str) -> None:
    assert type_v5(smiles).types[index] == expected
