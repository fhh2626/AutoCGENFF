"""Exact-map checks against the bundled CGenFF v5 RTF.

The predeclared correspondence is in SMILES atom order followed by appended H.
Only the scorer reads RTF types, *after* inference. Chemically equivalent H are
named in arbitrary fixed order. These are inspected regressions, not a holdout.
"""

import pytest

from autocgenff import bundled_force_field_dir
from autocgenff.topology import read_topology
from autocgenff.v5 import type_v5, v5_vocabulary

RTF = bundled_force_field_dir() / "top_all36_cgenff.rtf"
pytestmark = pytest.mark.skipif(not RTF.is_file(), reason="local v5 RTF not supplied")


@pytest.fixture(scope="module")
def reference():
    return read_topology(RTF)


def test_outputs_exactly_match_local_real_mass_vocabulary():
    # NG2D1's MASS omits the optional element field. Audit names, not the
    # narrower element map used by named-template identity verification.
    with RTF.open(encoding="utf-8", errors="replace") as handle:
        labels = {line.split()[2] for line in handle if line.startswith("MASS ")}
    assert len(labels) == 166
    assert v5_vocabulary() == labels - {"LPH", "OGTIP3", "HGTIP3"}


@pytest.mark.parametrize(
    ("name", "smiles", "names"),
    [
        (
            "23001K",
            "CC1=NOC2(CCCCC2)C1=O",
            "C01 C3 N10 O11 C12 C13 C14 C15 C16 C17 C2 O1 H01 H02 H03 H23 H24 H25 H26 H27 H28 H29 H30 H31 H32",
        ),
        (
            "23002G",
            "CC1=NN=C2CCC[C@@H]2CC1",
            "C02 C5 N6 N7 C8 C9 C10 C11 C12 C14 C15 H04 H05 H06 H21 H22 H23 H24 H25 H26 H13 H27 H28 H29 H30",
        ),
        ("CO2", "O=C=O", "OC1 C OC2"),
        ("CO3", "[O-]C(=O)[O-]", "O1 C1 O2 O3"),
        ("ALF4", "F[Al-](F)(F)F", "F1 AL1 F2 F3 F4"),
        ("FURA", "c1ccco1", "C1 C2 C3 C4 O5 H1 H2 H3 H4"),
        (
            "230020",
            "F[n+]1ccccc1-c1cccc[n+]1F",
            "F1 N2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 N13 F14 H15 H16 H17 H18 H19 H20 H21 H22",
        ),
        ("IODB", "c1ccccc1I", "C1 C2 C3 C4 C5 C6 I6 H1 H2 H3 H4 H5"),
        ("NIME", "C[N+](=O)[O-]", "C1 N2 O21 O22 H11 H12 H13"),
        ("DMSO", "CS(C)=O", "C3 S2 C7 O1 H4 H5 H6 H8 H9 H10"),
        ("ACEM", "CC(=O)N", "CC C O N HC1 HC2 HC3 HC HT"),
        ("SM169", "CNC(=S)N", "CD NE CZ SZ NH HD1 HD2 HD3 HE HH1 HH2"),
        (
            "BDFP",
            "c1ccc(C(F)(F)P(=O)([O-])O)cc1",
            "CG CD1 CE1 CZ CF F1 F2 P1 O3 O4 O2 CE2 CD2 HG HD1 HE1 H2 HE2 HD2",
        ),
        ("FETH", "CCF", "C1 C2 F21 H11 H12 H13 H22 H23"),
        ("DFET", "CC(F)F", "C1 C2 F22 F23 H11 H12 H13 H21"),
        ("TFET", "CC(F)(F)F", "C1 C2 F21 F22 F23 H11 H12 H13"),
        ("SM083", "CC(C)(F)C", "CA CB CG1 FG3 CG2 HA HA2 HA3 HG11 HG12 HG13 HG21 HG22 HG23"),
        ("BORO", "CB(O)O", "C1 B1 O1 O2 H11 H12 H13 HO1 HO2"),
        ("BONN", "CB([O-])[O-]", "C1 B1 O1 O2 H11 H12 H13"),
        (
            "BSEU",
            "Cn1c(=[Se])[nH]c(=O)c(C[NH2+]C)c1",
            "CM N1 C2 SE2 N3 C4 O4 C5 C N CA C6 HM1 HM2 HM3 H3 HC1 HC2 HN1 HN2 HA1 HA2 HA3 H6",
        ),
    ],
)
def test_local_rtf_atom_map(reference, name, smiles, names):
    result = type_v5(smiles)
    labels = {a.name: a.atom_type for a in reference.residues[name].atoms if a.atom_type != "LPH"}
    order = names.split()
    assert set(order) == labels.keys() and len(order) == len(labels)
    assert result.is_complete and result.types == tuple(labels[n] for n in order)
