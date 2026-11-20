"""Synthetic checks for the resonance-role and substituent rules added for B1.

Covers, with isolated structures rather than benchmark residues:

- a six-ring sultam (N-acylsulfonamide fused to benzene): the imide N is the
  sulfonamide type NG2S1, the aryl sulfonamide N-H keeps the aniline type
  NG311, and the carbonyl stays a plain amide CG2O1 instead of the
  aromatic-ring carbonyl CG2R63 (the B1 case family of 2300Q0/2300X9/230073);
- a five-ring sultam keeps the small-ring lactam pair NG2R53/CG2R53;
- a sulfonamide N-H keeps NG311 while a charged sulfamate arm is NG2S3;
- thione vs carbonyl inside an aromatic six-ring: =S marks the ring carbon
  CG2R63 but does not spread CG2R62, and its ortho H is HGR62;
- neutral amidine imine N stays NG2D1 instead of the cationic NG2P1.
"""

import pytest
from rdkit import Chem

from autocgenff.v5 import type_v5


def _types(smiles: str) -> list[str]:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    typed = type_v5(molecule)
    assert typed.is_complete
    return [atom.atom_type for atom in typed.atoms]


def test_six_ring_sultam_imide_types() -> None:
    types = _types("O=C1NS(=O)(=O)Nc2ccccc21")
    # O=C1 N S(=O)(=O) N aryl ... : carbonyl, imide N, sulfonyl S, sulfonyl O2, aniline N.
    assert types[0] == "OG2D1"
    assert types[1] == "CG2O1"
    assert types[2] == "NG2S1"
    assert types[3] == "SG3O2"
    assert types[4] == types[5] == "OG2P1"
    assert types[6] == "NG311"
    # Both sulfonamide N-H are plain polar hydrogens, not cationic ones.
    assert types[13] == types[14] == "HGP1"


def test_five_ring_sultam_keeps_lactam_pair() -> None:
    types = _types("O=C1NS(=O)(=O)c2ccccc21")
    assert types[1] == "CG2R53"
    assert types[2] == "NG2R53"


def test_sulfonamide_and_sulfamate_nh_boundary() -> None:
    # Sulfonamide N-H on an oxidized S stays the aniline type.
    assert _types("CNS(C)(=O)=O")[1] == "NG311"
    # An anionic sulfamate arm (single-bonded O-) keeps the sulfamate type.
    assert _types("CNS(=O)(=O)[O-]")[1] == "NG2S3"


def test_n_sulfonyl_urea_is_carbonate_not_amide() -> None:
    # A second single-bonded heteroatom neighbour makes the carbonyl a urea.
    assert _types("NC(=O)NS(=O)(=O)C")[1] == "CG2O6"


def test_thione_ring_carbon_and_ortho_hydrogen() -> None:
    types = _types("S=c1cccc[nH]1")
    # The C=S carbon is the aromatic-ring carbonyl subtype, but =S does not
    # spread CG2R62 around the ring the way =O does.
    assert types[0] == "SG2D1"
    assert types[1] == "CG2R63"
    assert types[2:6] == ["CG2R61"] * 4
    # Ortho H to the thione carbon (and to the ring N) is HGR62; meta/para stay HGR61.
    assert types[7] == "HGR62"   # C2 next to C1=S
    assert types[10] == "HGR62"  # C5 next to ring N
    assert types[8] == "HGR61"
    assert types[9] == "HGR61"
    assert types[11] == "HGP1"


def test_pyridone_ring_marks_carbonyl_neighbours() -> None:
    types = _types("O=c1cccc[nH]1")
    assert types[0] == "OG2D4"
    assert types[1] == "CG2R63"
    assert types[2:6] == ["CG2R62"] * 4
    # H on a CG2R62 carbon is HGR62 by carrier type.
    assert types[7:11] == ["HGR62"] * 4


@pytest.mark.parametrize("smiles", ["c1ccccc1", "CC(=O)c1ccccc1"])
def test_unactivated_ring_hydrogens_stay_hgr61(smiles: str) -> None:
    # An exocyclic C=O substituent carbon (ketone, ester) is not a C=X ring member.
    assert _types(smiles)[-5:] == ["HGR61"] * 5


def test_neutral_amidine_imine_is_ng2d1() -> None:
    types = _types("CC(=N)N")
    # The double-bonded N is the neutral imine, not a cationic amidinium N.
    assert types[2] == "NG2D1"
    assert types[3] == "NG321"


def test_four_ring_sultam_lactam_keeps_small_ring_type() -> None:
    # The N-sulfonyl-lactam leaf only applies outside the four/five-ring cases.
    assert _types("O=C1N(S(=O)(=O)C)CC1")[1] == "CG2R53"


def test_sulfonyl_on_phosphoramidate_and_thioamide_stays_specific() -> None:
    # The sulfonamide N-H leaf must not shadow the phosphoramidate or the
    # thioamide paths.
    assert _types("C[P](=O)(OC)NS(=O)(=O)C")[5] == "NG2S3"
    assert _types("CC(=S)NS(=O)(=O)C")[3] == "NG2S1"


def test_fusion_n_needs_conjugated_cation_edge() -> None:
    # A fusion N is the cationic ring type only when its in-ring neighbour is
    # double-bonded to an in-ring cation; a remote iminium separated by
    # saturated carbons stays NG2R51. (Positive cases: B1 23006H, 2300FB.)
    assert _types("C1C=[NH+]Cc2cccn21")[8] == "NG2R51"
