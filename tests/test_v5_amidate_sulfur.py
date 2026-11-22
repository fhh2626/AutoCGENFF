"""Regression cases for deprotonated amides and low-valent sulfur."""

from pathlib import Path

import pytest

from autocgenff import V5ForceField, parameterize_v5
from autocgenff.v5 import type_v5


@pytest.mark.parametrize("smiles", ["CC(=O)[N-]C", "CC([O-])=NC"])
def test_acyclic_amidate_charge_lives_on_nitrogen_in_both_drawings(smiles):
    atoms = type_v5(smiles).atoms
    assert [a.atom_type for a in atoms[:5]] == ["CG331", "CG2O1", "OG2D1", "NG2D1", "CG331"]
    assert [a.rule_charge for a in atoms[:5]] == [0, 0, 0, -1, 0]
    assert sum(a.rule_charge for a in atoms) == sum(a.formal_charge for a in atoms) == -1
    assert [a.index for a in atoms if a.improper_center] == [1]


@pytest.mark.parametrize(
    "smiles, index, expected_type, seed",
    [
        ("CCS[O-]", 3, "OG312", -1),
        ("CCSS", 3, "SG311", 0),
        ("CCSSCC", 3, "SG301", 0),
        ("CCSOO", 3, "OG301", 0),
        ("CCSN=O", 4, "OG2D1", 0),
        ("CS(=O)(=O)[O-]", 4, "OG2P1", 0),
        ("COS(=O)(=O)[O-]", 1, "OG303", 0),
        ("CC[N+](=O)[O-]", 3, "OG2N1", 0),
        ("CC(=O)NC", 3, "NG2S1", 0),
    ],
)
def test_specific_rules_leave_neighboring_chemistry_distinct(smiles, index, expected_type, seed):
    atoms = type_v5(smiles).atoms
    assert (atoms[index].atom_type, atoms[index].rule_charge) == (expected_type, seed)
    assert sum(a.rule_charge for a in atoms) == sum(a.formal_charge for a in atoms)


def test_imide_anion_is_not_the_monoamide_rule():
    nitrogen = type_v5("O=C1CCC(=O)[N-]1").atoms[6]
    assert nitrogen.atom_type == "NG2R53" and nitrogen.rule_charge == 0


@pytest.mark.parametrize("smiles", ["CN=C=S", "CN=C=O", "O=C=O", "N=C=N", "CN=C=NC", "CC=C=NC"])
def test_two_coordinate_cumulene_has_no_improper_center(smiles):
    assert not any(atom.improper_center for atom in type_v5(smiles).atoms)


def test_three_coordinate_amidine_still_has_an_improper_center():
    result = type_v5("CC(=N)N")
    assert result.atoms[1].improper_center


def test_sulfenate_resolves_native_increments():
    root = Path(__file__).resolve().parents[1]
    data = root / "force_fields"
    required = ("par_all36_cgenff.prm", "top_all36_cgenff.rtf", "par_all36_cgenff_increments.prm")
    if not all((data / name).is_file() for name in required):
        pytest.skip("requires local native v5 force-field files")
    ff = V5ForceField.from_directory(data)
    result = parameterize_v5("CCS[O-]", ff)
    assert result.is_complete
    assert sum(a.partial_charge for a in result.atoms) == pytest.approx(-1)
