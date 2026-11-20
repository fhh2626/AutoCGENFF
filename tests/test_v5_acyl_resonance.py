"""Acyl resonance equivalence and scope of ring-perimeter matching."""

import pytest
from rdkit import Chem, rdBase

from autocgenff.charges import formal_baseline
from autocgenff.molecule import prepare_molecule
from autocgenff.v5 import _prepare_v5, type_v5
from autocgenff.v5._engine import type_perception
from autocgenff.v5._perception import perceive
from autocgenff.v5._rules import parse


def _indexed_edges(molecule: Chem.Mol) -> tuple[tuple[int, int], ...]:
    return tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds())


@pytest.mark.parametrize(
    ("neutral", "separated"),
    [
        ("O=C(C)NC", "[O-]C(C)=[NH+]C"),
        ("O=C(C)OC", "[O-]C(C)=[O+]C"),
        ("O=c1cccc[nH]1", "[O-]c1cccc[nH+]1"),
        ("O=C(C)NCC[NH3+].[Cl-]", "[O-]C(C)=[NH+]CC[NH3+].[Cl-]"),
        ("O=C(C)NCCNC(C)=O", "[O-]C(C)=[NH+]CC[NH+]=C(C)[O-]"),
    ],
)
def test_local_acyl_contributors_preserve_indexes_and_component_charge(
    neutral: str, separated: str
) -> None:
    original = prepare_molecule(separated)
    prepared = _prepare_v5(original)
    assert _indexed_edges(prepared) == _indexed_edges(original)
    assert [a.GetAtomicNum() for a in prepared.GetAtoms()] == [
        a.GetAtomicNum() for a in original.GetAtoms()
    ]
    assert prepared.GetAtomWithIdx(0).GetFormalCharge() == 0
    assert original.GetAtomWithIdx(0).GetFormalCharge() == -1
    for component in Chem.GetMolFrags(original):
        assert sum(original.GetAtomWithIdx(i).GetFormalCharge() for i in component) == sum(
            prepared.GetAtomWithIdx(i).GetFormalCharge() for i in component
        )
    typed = type_v5(original)
    assert typed == type_v5(neutral)
    kekule = _prepare_v5(original)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    formal_baseline(kekule, tuple(a.rule_charge for a in typed.atoms))


@pytest.mark.parametrize(
    "smiles",
    [
        "CC=[NH+]C.[Cl-]",  # iminium and counterion
        "CC(=O)C=[NH+]C",  # acyliminium, not an acyl pi-pair
        "CC(=O)[O-]",  # carboxylate
        "CC(=O)N[O-]",  # deprotonated hydroxamate
        "[O-]C(C)=[N+](C)OC",  # N–O-bearing donor outside the simple amide rule
        "CC(=N[O-])C",  # oximate
        "C[N+](C)(C)[O-]",  # N-oxide
        "C[N+](=O)[O-]",  # nitro
        "O=C1NC=NC=[NH+]1",  # cationic aza aromatic, not an acyl pi-pair
        "O=C1[N-]C=NC=C1",  # N-localized pyrimidinolate
        "[O-]C1=NC=NC=C1",  # O-localized pyrimidinolate
        "[O-]c1cccc([O-])[nH+]1",  # competing acyl acceptors for one donor
        "[O-]CCC[n+]1ccccc1",  # remote oxide and aza-aromatic cation
        "O=C1[N-]C(OC)=NC=C1",  # alkoxy pyrimidinolate
        r"[O-]/C(C)=[N+](\C)CC",  # specified stereo on the changing double bond
    ],
)
def test_out_of_scope_graphs_are_not_rewritten(smiles: str) -> None:
    assert Chem.MolToSmiles(_prepare_v5(smiles)) == Chem.MolToSmiles(prepare_molecule(smiles))


def test_competing_acyl_normalizations_do_not_depend_on_atom_order() -> None:
    original = prepare_molecule("[O-]c1[nH+]c(C)c([O-])[nH+]c1")
    with rdBase.BlockLogs():
        expected = _prepare_v5(original)
        baseline = type_v5(original)
    assert Chem.MolToSmiles(expected) == Chem.MolToSmiles(original)
    order = list(reversed(range(original.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(original, order)
    with rdBase.BlockLogs():
        prepared = _prepare_v5(renumbered)
        assigned = type_v5(renumbered)
    assert Chem.MolToSmiles(prepared) == Chem.MolToSmiles(expected)
    for atom, old in zip(assigned.atoms, order, strict=True):
        reference = baseline.atoms[old]
        assert (atom.atom_type, atom.formal_charge, atom.rule_charge, atom.improper_center) == (
            reference.atom_type, reference.formal_charge, reference.rule_charge,
            reference.improper_center,
        )
    for bond in prepared.GetBonds():
        original_bond = expected.GetBondBetweenAtoms(
            order[bond.GetBeginAtomIdx()], order[bond.GetEndAtomIdx()],
        )
        assert bond.GetBondType() == original_bond.GetBondType()


def test_ring_neighbors_try_both_assignments_for_overlapping_groups() -> None:
    rules = parse("cat main\ntyp Hit : rne ring 3 ( ! ( el H ) ) ( el C )\ntyp Miss :\nend\n")
    molecule = prepare_molecule("C1CO1")
    reversed_ring = Chem.RenumberAtoms(molecule, [0, 2, 1, *range(3, molecule.GetNumAtoms())])
    for graph in (molecule, reversed_ring):
        assert type_perception(rules, perceive(graph))[0].type == "Hit"


def test_resonance_normalization_preserves_nonlocal_stereo_and_input_metadata() -> None:
    original = prepare_molecule(r"[18O-]C(C)=[NH+]C[C@H](F)/C=C/Cl")
    original.SetProp("_Name", "synthetic acyl")
    original.GetAtomWithIdx(0).SetProp("atomLabel", "oxide")
    original.GetBondWithIdx(0).SetProp("source", "annotated")
    conformer = Chem.Conformer(original.GetNumAtoms())
    conformer.SetAtomPosition(0, (1.0, 2.0, 3.0))
    original.AddConformer(conformer)
    result = _prepare_v5(original)
    assert result.GetProp("_Name") == "synthetic acyl"
    assert result.GetAtomWithIdx(0).GetProp("atomLabel") == "oxide"
    assert result.GetAtomWithIdx(0).GetIsotope() == 18
    assert result.GetBondWithIdx(0).GetProp("source") == "annotated"
    assert tuple(result.GetConformer().GetAtomPosition(0)) == (1.0, 2.0, 3.0)
    assert Chem.MolToSmiles(Chem.RemoveHs(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(r"[18O]=C(C)NC[C@H](F)/C=C/Cl")
    )
    assert original.GetBondWithIdx(0).GetBondType() == Chem.BondType.SINGLE


def test_acyl_normalization_refreshes_cip_without_inverting_tetrahedral_stereo() -> None:
    original = prepare_molecule("[O-]C(=[NH+]C)[C@H](C(=O)C)F")
    neutral = _prepare_v5("O=C(NC)[C@H](C(=O)C)F")
    result = _prepare_v5(original)
    before, expected, after = (mol.GetAtomWithIdx(4) for mol in (original, neutral, result))
    assert before.GetProp("_CIPCode") == "R"
    assert after.GetProp("_CIPCode") == expected.GetProp("_CIPCode") == "S"
    assert before.GetChiralTag() == after.GetChiralTag() == expected.GetChiralTag()
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(neutral)


def test_acyl_normalization_refreshes_alkene_priorities_without_inverting_stereo() -> None:
    original = prepare_molecule(r"[O-]C(=[NH+]C)/C(C(C)=O)=C/F")
    neutral = _prepare_v5(r"O=C(NC)/C(C(C)=O)=C/F")
    result = _prepare_v5(original)
    before, expected, after = (mol.GetBondWithIdx(7) for mol in (original, neutral, result))
    assert before.GetStereo() == Chem.BondStereo.STEREOZ
    assert tuple(before.GetStereoAtoms()) == (5, 9)
    assert after.GetStereo() == expected.GetStereo() == Chem.BondStereo.STEREOE
    assert tuple(after.GetStereoAtoms()) == tuple(expected.GetStereoAtoms()) == (1, 9)
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(neutral)
