from math import fsum

import pytest
from rdkit import Chem

from autocgenff.analogy import PenaltyModel
from autocgenff.charges import ChargeModel, assign_charges, formal_baseline
from autocgenff.errors import AutoCGENFFError
from autocgenff.fitting import TypedReference, fit_increments
from autocgenff.molecule import Kind, Term, inventory, prepare_molecule


def test_linear_center_types_cannot_be_a_bare_string() -> None:
    with pytest.raises(AutoCGENFFError, match="collection of explicit atom types"):
        ChargeModel({}, linear_centers="CG1N1")


def test_bond_angle_proper_increment_reversal_and_conservation():
    molecule = Chem.MolFromSmiles("CCCC")
    model = ChargeModel(
        {("A", "B"): (0.2,), ("A", "B", "C"): (0.3, 0.5), ("A", "B", "C", "D"): (0.4, 0.7, 0.6)}
    )
    paths = (Term(Kind.BOND, (0, 1)), Term(Kind.ANGLE, (0, 1, 2)), Term(Kind.PROPER, (0, 1, 2, 3)))
    forward = assign_charges(molecule, ("A", "B", "C", "D"), paths, model)
    reverse = assign_charges(
        molecule,
        ("A", "B", "C", "D"),
        tuple(Term(path.kind, path.atoms[::-1]) for path in paths),
        model,
    )
    assert forward.charges == pytest.approx((-0.9, -0.3, 0.6, 0.6))
    assert reverse.charges == pytest.approx(forward.charges)
    assert fsum(forward.charges) == pytest.approx(0)
    assert reverse.terms[-1].values == (-0.6, -0.7, -0.4)


def test_palindromic_inputs_require_antisymmetric_increment():
    molecule = Chem.MolFromSmiles("CCC")
    term = Term(Kind.ANGLE, (0, 1, 2))
    with pytest.raises(AutoCGENFFError, match="Palindromic"):
        ChargeModel({("A", "B", "A"): (0.1, 0.1)})
    model = ChargeModel({("A", "B", "A"): (0.1, -0.1)})
    charges = assign_charges(molecule, ("A", "B", "A"), [term], model)
    assert charges.charges == pytest.approx((-0.1, 0.2, -0.1))


def test_missing_increment_returns_no_partial_charges():
    molecule = Chem.MolFromSmiles("CO")
    charges = assign_charges(molecule, ("A", "B"), inventory(molecule), ChargeModel({}))
    assert charges.charges is None and charges.atom_penalties is None
    assert charges.terms[0].status == "missing"


def test_linear_centers_apply_after_exact_rows_and_only_to_central_proper_atoms():
    molecule = Chem.MolFromSmiles("CCCC")
    path = [Term(Kind.PROPER, (0, 1, 2, 3))]
    with pytest.raises(AutoCGENFFError, match="Linear centers"):
        ChargeModel({}, linear_centers=frozenset({"X"}))
    model = ChargeModel({("A", "L", "C", "D"): (0.1, 0.2, 0.3)}, linear_centers=frozenset({"L"}))
    exact = assign_charges(molecule, ("A", "L", "C", "D"), path, model)
    assert exact.terms[0].status == "exact" and exact.charges == pytest.approx((-0.1, -0.1, -0.1, 0.3))
    linear = assign_charges(molecule, ("B", "C", "L", "E"), path, model)
    assert linear.terms[0].status == "linear_not_applicable" and linear.charges == (0, 0, 0, 0)
    terminal = assign_charges(molecule, ("L", "B", "C", "E"), path, model)
    assert terminal.terms[0].status == "missing" and terminal.charges is None


def test_cutoff_returns_zero_transfers_with_nonzero_atom_penalties():
    molecule = Chem.MolFromSmiles("CCCC")
    model = ChargeModel({("A", "Q", "C", "D"): (0.1, 0.2, 0.3)})
    penalties = PenaltyModel(nonbonded={("B", "Q"): 6})
    result = assign_charges(
        molecule,
        ("A", "B", "C", "D"),
        [Term(Kind.PROPER, (0, 1, 2, 3))],
        model,
        penalties=penalties,
    )
    assert result.charges == (0.0,) * 4
    assert result.terms[0].status == "cutoff" and result.terms[0].penalty == 50
    assert result.atom_penalties == pytest.approx((2.5,) * 4)


def test_custom_baseline_preserves_charge_on_each_fragment():
    molecule = prepare_molecule("[Na+].[Cl-]")
    with pytest.raises(AutoCGENFFError, match="preserve formal charge"):
        formal_baseline(molecule, (0.0, 0.0))
    assert formal_baseline(molecule) == (1, -1)


def test_staged_fitting_recovers_known_bond_transfer():
    molecule = prepare_molecule("ClCl")
    references = [TypedReference(molecule, ("LEFT", "RIGHT"), (-0.2, 0.2))]
    result = fit_increments(references, ridge=0)
    assert result.model.increments[("LEFT", "RIGHT")] == pytest.approx((0.2,))
    assert [stage.order for stage in result.stages] == [2, 3, 4]
    assert result.stages[0].rms_error < 1e-12
    with pytest.raises(AutoCGENFFError, match="target charge total"):
        fit_increments([TypedReference(molecule, ("LEFT", "RIGHT"), (-0.2, 0.35))])
