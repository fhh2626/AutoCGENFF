"""Synthetic checks for consumers of aromatic seven-ring carbon types."""

from pathlib import Path
from random import Random

import pytest
from rdkit import Chem

from autocgenff import AutoCGENFFError, parameterize_v5
from autocgenff.molecule import Kind, prepare_molecule
from autocgenff.v5 import type_v5

# These neutral controls have two carbonyl carbons that reach the same CG2R71
# leaf. One is an acyl derivative; the other is a ketone in the same ring.
FUSED_LACTAM = "O=C1N(C)C=Cc2occc2C1=O"
FUSED_LACTONE = "O=C1OC=Cc2sccc2C1=O"


@pytest.mark.parametrize("smiles", [FUSED_LACTAM, FUSED_LACTONE])
def test_troponoid_oxygen_requires_ketone_connectivity(smiles: str) -> None:
    result = type_v5(smiles)
    acyl_o, acyl_c, *_, ketone_c, ketone_o = (a for a in result.atoms if a.element != "H")
    assert result.is_complete
    assert (acyl_c.atom_type, ketone_c.atom_type) == ("CG2R71", "CG2R71")
    assert (acyl_o.atom_type, ketone_o.atom_type) == ("OG2D1", "OG2D3")
    assert all(a.formal_charge == a.rule_charge == 0 for a in result.atoms)
    assert not any(a.improper_center for a in result.atoms)


def test_tropone_and_nonaromatic_lactam_keep_local_annotations() -> None:
    tropone = type_v5("O=C1C=CC=CC=C1")
    assert tropone.types[:2] == ("OG2D3", "CG2O5")
    assert tropone.atoms[1].improper_center
    assert tropone.types.count("HGR71") == 6
    lactam = type_v5("O=C1NCC=CC=C1")
    assert lactam.types[:2] == ("OG2D1", "CG2O1")
    assert lactam.atoms[1].improper_center
    assert "HGR71" not in lactam.types


@pytest.mark.parametrize(
    ("smiles", "aromatic_seven"),
    [
        ("c1ccc2cccc-2cc1", True),  # azulene: both fusion carbons use CG2RC7
        ("O=C1NCCCc2occc21", False),  # five-ring fusion next to a lactam C=O
        ("C1=CCCCc2sccc21", False),  # five-ring fusion next to a localized C=C
        ("C1CCCCc2sccc21", False),  # the corresponding saturated near-control
        ("O=C1NCCCc2n[nH]cc21", False),  # five-ring aza fusion
    ],
)
def test_aromatic_seven_ring_types_are_not_borrowed_by_nonaromatic_fusions(
    smiles: str, aromatic_seven: bool
) -> None:
    original = prepare_molecule(smiles)
    typed = type_v5(original)
    assert typed.is_complete
    if aromatic_seven:
        assert typed.types.count("CG2RC7") == 2
        assert typed.types.count("CG2R71") == typed.types.count("HGR71") == 5
    else:
        # A scope check, not a claim that all remaining local types are validated.
        assert {"CG2RC7", "CG2R71", "HGR71"}.isdisjoint(typed.types)
    order = list(reversed(range(original.GetNumAtoms())))
    rng = Random(13)
    for _ in range(8):
        reassigned = type_v5(Chem.RenumberAtoms(original, order))
        expected = (typed.atoms[i] for i in order)
        for atom, old in zip(reassigned.atoms, expected, strict=True):
            assert (atom.atom_type, atom.formal_charge, atom.rule_charge, atom.improper_center) == (
                old.atom_type, old.formal_charge, old.rule_charge, old.improper_center,
            )
        rng.shuffle(order)


def test_fused_aza_nitrogen_does_not_borrow_nitro_type() -> None:
    quinoline = type_v5("c1ccc2ncccc2c1")
    nitrobenzene = type_v5("O=[N+]([O-])c1ccccc1")
    assert quinoline.is_complete and nitrobenzene.is_complete
    assert "NG2O1" not in quinoline.types
    assert nitrobenzene.types[1] == "NG2O1"


def test_oxygen_changes_mechanical_and_increment_selection_without_inventing_coverage(
    tmp_path: Path,
) -> None:
    (tmp_path / "top_all36_cgenff.rtf").write_text(
        "MASS -1 CG2R71 12 C\nMASS -1 OG2D1 16 O\nMASS -1 OG2D3 16 O\nRESI SYN 0\nEND\n",
        encoding="ascii",
    )
    # Original synthetic values. Parameter selection, not physical validation.
    (tmp_path / "par_all36_cgenff.prm").write_text(
        "BONDS\nOG2D1 CG2R71 11 1.1\nOG2D3 CG2R71 12 1.2\nEND\n", encoding="ascii"
    )
    (tmp_path / "par_all36_cgenff_increments.prm").write_text(
        "BONDS\nOG2D1 CG2R71 11 1.1 !<% .1 %>!\nOG2D3 CG2R71 12 1.2 !<% .2 %>!\nEND\n",
        encoding="ascii",
    )
    result = parameterize_v5(FUSED_LACTAM, tmp_path)
    for path, oxygen, force_constant, increment in [
        ((0, 1), "OG2D1", 11, 0.1),
        ((11, 12), "OG2D3", 12, -0.2),
    ]:
        bond = next(t for t in result.terms if t.term.kind == Kind.BOND and t.term.atoms == path)
        charge = next(t for t in result.charge_diagnostics.terms if t.term == bond.term)
        assert bond.status == charge.status == "exact"
        assert bond.match is not None and bond.match.records[0].values[0] == force_constant
        assert charge.source_types == (oxygen, "CG2R71") and charge.values == (increment,)
    assert not any(t.term.kind == Kind.IMPROPER for t in result.terms)
    assert not result.is_complete and result.charge_diagnostics.charges is None


@pytest.mark.parametrize(
    ("neutral", "charge_separated"),
    [
        (FUSED_LACTAM, "[O-]C1=[N+](C)C=Cc2occc2C1=O"),
        (FUSED_LACTONE, "[O-]C1=[O+]C=Cc2sccc2C1=O"),
    ],
)
def test_charge_separated_acyl_drawings_use_the_same_typing_and_parameters(
    neutral: str, charge_separated: str, tmp_path: Path
) -> None:
    source = prepare_molecule(charge_separated)
    typed = type_v5(source)
    assert typed == type_v5(neutral)
    assert typed.is_complete
    assert all(a.formal_charge == a.rule_charge == 0 for a in typed.atoms)
    assert typed.types[:2] == ("OG2D1", "CG2R71")
    assert [a.atom_type for a in typed.atoms if a.element != "H"][-1] == "OG2D3"
    assert not any(a.improper_center for a in typed.atoms)

    (tmp_path / "top_all36_cgenff.rtf").write_text(
        "MASS -1 CG2R71 12 C\nMASS -1 OG2D1 16 O\nRESI SYN 0\nEND\n", encoding="ascii"
    )
    (tmp_path / "par_all36_cgenff.prm").write_text(
        "BONDS\nOG2D1 CG2R71 11 1.1\nEND\n", encoding="ascii"
    )
    (tmp_path / "par_all36_cgenff_increments.prm").write_text(
        "BONDS\nOG2D1 CG2R71 11 1.1 !<% .1 %>!\nEND\n", encoding="ascii"
    )
    assignment = parameterize_v5(source, tmp_path)
    assert assignment == parameterize_v5(neutral, tmp_path)
    assert tuple(a.atom_type for a in assignment.atoms) == typed.types
    # Both APIs expose the normalized charges, not the unchanged input pair.
    assert [source.GetAtomWithIdx(i).GetFormalCharge() for i in (0, 2)] == [-1, 1]
    for atom, typed_atom in zip(assignment.atoms, typed.atoms, strict=True):
        assert atom.formal_charge == typed_atom.formal_charge == 0
        assert atom.rule_charge == typed_atom.rule_charge == 0
        assert atom.partial_charge is None
    assert assignment.bonds[0].order == 2
    # The partial bundle demonstrates lookup, not full coverage.
    assert assignment.terms[0].status == "exact"
    assert not assignment.is_complete


def test_oxygen_guard_does_not_bypass_unbalanced_charge_seed(tmp_path: Path) -> None:
    cation = "O=C1N(C)C=Cc2occc2[CH+]1"
    assert type_v5(cation).types[0] == "OG2D1"
    with pytest.raises(AutoCGENFFError, match="preserve formal charge"):
        parameterize_v5(cation, tmp_path)
