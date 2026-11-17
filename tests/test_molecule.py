from pathlib import Path

import pytest
from rdkit import Chem

from autocgenff.errors import AutoCGENFFError
from autocgenff.molecule import Kind, inventory, prepare_molecule
from autocgenff.typing import TypingRule, type_atoms


def test_preserves_existing_hydrogens_and_does_not_mutate_mol():
    params = Chem.SmilesParserParams()
    params.removeHs = False
    source = Chem.MolFromSmiles("[H]OC", params)
    prepared = prepare_molecule(source)
    assert [atom.GetSymbol() for atom in prepared.GetAtoms()] == ["H", "O", "C", "H", "H", "H"]
    assert source.GetNumAtoms() == 3


def test_sdf_keeps_input_order_and_rejects_multiple_records(tmp_path: Path):
    path = tmp_path / "mol.sdf"
    molecule = prepare_molecule("OCC")
    with Chem.SDWriter(str(path)) as writer:
        writer.write(molecule)
    assert [a.GetSymbol() for a in prepare_molecule(path).GetAtoms()][:3] == ["O", "C", "C"]
    with Chem.SDWriter(str(path)) as writer:
        writer.write(molecule)
        writer.write(molecule)
    with pytest.raises(AutoCGENFFError, match="one SDF"):
        prepare_molecule(path)


def test_mol2_partial_charge_is_not_a_formal_charge(tmp_path: Path):
    path = tmp_path / "mol.mol2"
    path.write_text(
        """@<TRIPOS>MOLECULE
water
3 2 0 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
1 O1 0.0 0.0 0.0 O.3 1 WAT -0.8
2 H1 1.0 0.0 0.0 H 1 WAT 0.4
3 H2 0.0 1.0 0.0 H 1 WAT 0.4
@<TRIPOS>BOND
1 1 2 1
2 1 3 1
""",
        encoding="utf-8",
    )
    assert [atom.GetFormalCharge() for atom in prepare_molecule(path).GetAtoms()] == [0, 0, 0]


def test_builtins_and_prioritized_user_extension():
    molecule = prepare_molecule("CCO")
    extension = TypingRule("custom", "[#8]-[#6]", "CUSTOM", priority=100, target=1)
    types = type_atoms(molecule, (extension,))
    assert [item.atom_type for item in types[:3]] == ["CG331", "CUSTOM", "OG311"]
    assert types[1].source == "custom"
    with pytest.raises(AutoCGENFFError, match="No atom typing rule"):
        type_atoms(prepare_molecule("CCl"))
    with pytest.raises(AutoCGENFFError, match="Invalid SMARTS or target"):
        TypingRule("bad", "[C]", "CUSTOM", target=2)


def test_symmetrical_query_types_all_target_atoms():
    assert [atom.atom_type for atom in type_atoms(prepare_molecule("c1ccccc1"))] == [
        *(["CG2R61"] * 6),
        *(["HGR61"] * 6),
    ]


def test_inventory_uses_unique_propers_and_explicit_improper_dependencies():
    molecule = prepare_molecule("CCO")
    terms = inventory(molecule, ((0, 1, 3, 4),))
    assert sum(term.kind == Kind.BOND for term in terms) == 8
    assert sum(term.kind == Kind.ANGLE for term in terms) == 13
    assert sum(term.kind == Kind.PROPER for term in terms) == 12
    assert terms[-1].atoms == (0, 1, 3, 4)
    with pytest.raises(AutoCGENFFError, match="Invalid declared improper"):
        inventory(molecule, ((0, 1, 2, 3),))
