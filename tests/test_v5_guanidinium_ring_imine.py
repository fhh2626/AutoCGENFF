"""Synthetic tests of the neutral five-ring imine role and its scope."""

from pathlib import Path

import pytest
from rdkit import Chem

from autocgenff import AutoCGENFFError, parameterize_v5
from autocgenff.analogy import PenaltyModel
from autocgenff.assignment import _lookup_term
from autocgenff.charges import assign_charges
from autocgenff.molecule import Kind, Term
from autocgenff.parameters import read_parameters
from autocgenff.v5 import _profile, type_v5
from autocgenff.v5._engine import type_perception
from autocgenff.v5._perception import perceive
from autocgenff.v5._rules import RuleSet, parse
from autocgenff.v5_inputs import read_native_increments


@pytest.mark.parametrize(
    ("cumulative", "neutral", "center", "amine", "category"),
    [
        ("CNC1=[N+]=C(C)CN1C", "C[NH+]=C1N=C(C)CN1C", 2, 1, "NG2D"),
        ("NC1=[N+]=CC(=C)N1", "[NH2+]=C1N=CC(=C)N1", 1, 0, "NG2R5"),
        ("NC1=[N+]=CC2CCCN12", "[NH2+]=C1N=CC2CCCN12", 1, 0, "NG2D"),
    ],
)
def test_role_in_both_drawings_changes_only_target_type(
    cumulative: str, neutral: str, center: int, amine: int, category: str
) -> None:
    predicate = ("guanidinium_ring_imine",)
    legacy = RuleSet({
        cat: [rule for rule in rules if predicate not in rule.conditions]
        for cat, rules in _profile().categories.items()
    })
    target = center + 1
    for smiles, source in ((cumulative, target), (neutral, amine)):
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        order = list(range(molecule.GetNumAtoms()))
        for permutation in (order, order[::-1], order[1::2] + order[::2]):
            reordered = Chem.RenumberAtoms(molecule, permutation)
            original_graph = Chem.MolToMolBlock(reordered)
            typed = type_v5(reordered)
            assert Chem.MolToMolBlock(reordered) == original_graph
            assert typed.is_complete
            imine = typed.atoms[permutation.index(target)]
            assert imine.atom_type == "NG2R50"
            assert imine.trace[-1].startswith(f"{category}:")
            assert _profile().categories[category][0].conditions == [predicate]
            assert imine.trace[-1] == str(_profile().categories[category][0])
            assert [a.formal_charge for a in typed.atoms] == [int(i == source) for i in permutation]
            assert [a.rule_charge for a in typed.atoms] == [int(i == center) for i in permutation]
            before = type_perception(legacy, perceive(reordered))
            for a, b in zip(typed.atoms, before, strict=True):
                assert (a.rule_charge, a.improper_center) == (b.rule_charge, b.improper_center)
                if a.index != imine.index:
                    assert a.atom_type == b.type
                    assert a.errors == tuple(b.errors)
                    assert a.warnings == tuple(b.warnings)
                elif source == target or category == "NG2D":
                    assert b.type == "NG2R52"


@pytest.mark.parametrize(
    ("smiles", "target", "expected", "net_seed"),
    [
        ("CC1=[NH+]CCC1", 2, "NG2R52", 1),  # ordinary iminium
        ("CC1=NCCC1", 2, "NG2R50", 0),      # ordinary neutral imine uses the old leaf
        ("NC(N)=[N+]=C(C)C", 3, "NG2P1", 1),  # no five-ring
        ("CN=[N+]=[N-]", 2, "NG1T1", 0),   # azide precedence
        ("c1cc[nH+]cc1", 3, "NG2R61", 1),  # aromatic six-ring precedence
        ("CC1=[N+]=CCN1", 2, "NG2R52", 2),  # amidinium, not guanidinium
        ("CC1=NCC(C[NH3+])C1", 2, "NG2R50", 1),  # remote positive charge
        ("CC(=O)NC1=[N+]=CCN1S(=O)(=O)C", 5, "NG2R52", 1),  # both lone pairs committed
        ("C[N+](C)(C)C1=[N+]=CCN1", 5, "NG2R52", 2),  # not a neutral amino flank
        ("NC1=[N+]=C(N)NN1", 2, "NG2R52", 2),  # two guanidinium ends
        ("NC1=[N+]=CC[NH+]1C", 2, "NG2R52", 2),  # independent charge in the ring
    ],
)
def test_other_chemistry_keeps_legacy_leaf_and_seeds(
    smiles: str, target: int, expected: str, net_seed: int
) -> None:
    typed = type_v5(smiles)
    atom = typed.atoms[target]
    assert atom.atom_type == expected
    assert atom.trace[-1] not in {str(_profile().categories[cat][0]) for cat in ("NG2D", "NG2R5")}
    assert sum(a.rule_charge for a in typed.atoms) == net_seed
    if sum(a.formal_charge for a in typed.atoms) != net_seed:
        with pytest.raises(AutoCGENFFError, match="preserve formal charge"):
            parameterize_v5(smiles, Path("absent"))


@pytest.mark.parametrize("smiles", [
    "N#CC=CNC1=[N+]=CCN1S(=O)(=O)C",
    "N#CC=C[NH+]=C1N=CCN1S(=O)(=O)C",
])
def test_conjugated_flanks_do_not_supply_a_donor_witness(smiles: str) -> None:
    typed = type_v5(smiles)
    imine = typed.atoms[6]
    assert imine.atom_type == "NG2R52"
    assert imine.trace[-1] != str(_profile().categories["NG2D"][0])
    assert sum(a.rule_charge for a in typed.atoms) == sum(a.formal_charge for a in typed.atoms) == 1


@pytest.mark.parametrize(("smiles", "source", "center"), [
    ("NC1=[N+]=C(N)CN1", 2, 1),
    ("[NH2+]=C1N=C(N)CN1", 0, 1),
    ("NC1=NC(=[NH2+])CN1", 4, 3),
])
def test_competing_amidinium_keeps_legacy_ring_imine_role(
    smiles: str, source: int, center: int
) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    order = list(range(molecule.GetNumAtoms()))
    override = {str(_profile().categories[cat][0]) for cat in ("NG2D", "NG2R5")}
    for permutation in (order, order[::-1]):
        typed = type_v5(Chem.RenumberAtoms(molecule, permutation))
        assert typed.is_complete
        imine = typed.atoms[permutation.index(2)]
        assert imine.atom_type == "NG2R52"
        assert imine.trace[-1] not in override
        assert [a.formal_charge for a in typed.atoms] == [int(i == source) for i in permutation]
        assert [a.rule_charge for a in typed.atoms] == [int(i == center) for i in permutation]


def test_role_requires_an_existing_center_seed_attributable_to_the_drawn_source() -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("NC1=[N+]=CCN1"))
    perception = perceive(molecule)
    for action, expected in (("iminium_charge", "ROLE"), ("charge 1", "OTHER"), ("", "OTHER")):
        rules = parse(f"""
        cat main
        typ ROLE : guanidinium_ring_imine
        typ CEN  : el C ne ( el N bo 2 fc 1 ) ( el N bo 1 ) ( el N bo 1 ) {action}
        typ OTHER :
        end
        """)
        assert type_perception(rules, perception)[2].type == expected


@pytest.mark.parametrize("status", ["exact", "analogy", "missing"])
def test_native_lookup_keeps_exact_analogy_and_missing_distinct(tmp_path: Path, status: str) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("NC1=[N+]=CCN1"))
    typed = type_v5(molecule)
    assert all(typed.types)
    # One actual bond path from the typed motif, with synthetic PRM records.
    term = Term(Kind.BOND, (1, 2))
    key = tuple(typed.types[i] for i in term.atoms)
    assert key == ("CG2R53", "NG2R50")
    row = "CG2R53 NG2R50" if status == "exact" else "CGALT NG2R50"
    path = tmp_path / "synthetic.prm"
    path.write_text(f"BONDS\n{row} 300 1.3 !<% .1 %>!\nEND\n", encoding="utf-8")
    substitutions = {("CG2R53", "CGALT"): 0.1} if status == "analogy" else {}
    penalties = PenaltyModel(bonded=substitutions, nonbonded=substitutions)
    mechanical, _, match = _lookup_term(read_parameters(path), Kind.BOND, key, penalties, None)
    assert mechanical == status
    assert (match is None) == (status == "missing")
    baseline = tuple(a.rule_charge for a in typed.atoms)
    charges = assign_charges(
        molecule, typed.types, [term], read_native_increments(path),
        penalties=penalties, baseline=baseline,
    )
    assert charges.terms[0].status == status
    if status == "missing":
        assert charges.charges is None
    else:
        assert charges.terms[0].source_types == tuple(row.split())
        assert charges.charges == pytest.approx((0, 0.9, 0.1, *baseline[3:]))
