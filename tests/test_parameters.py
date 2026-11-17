from pathlib import Path

import pytest

from autocgenff.analogy import BondGroup, PenaltyModel, best_analogy
from autocgenff.errors import AutoCGENFFError
from autocgenff.molecule import Kind
from autocgenff.parameters import read_parameters


def test_prm_reversal_wildcard_priority_and_multiterm(tmp_path: Path):
    path = tmp_path / "example.prm"
    path.write_text(
        """* Original synthetic fixture
BONDS
A B 100 1.5
ANGLES
A B C 12 105 4 1.8
DIHEDRALS
X B C X 6 1 0
A B C D 2 1 180
A B C D 1 3 0
IMPROPERS
B X X A 10 0 0
B C C A 11 0 0
NONBONDED nbxmod 5 -
cutnb 13.5
A 0 -0.12 1.8 0 -0.045 1.5
B 0 -0.02 1.3
NBFIX
A B -0.15 3.0 -0.1 2.9
END
""",
        encoding="utf-8",
    )
    prm = read_parameters(path)
    bond = prm.match(Kind.BOND, ("B", "A"))
    assert bond.reversed and bond.records[0].values == (100, 1.5)
    assert prm.match(Kind.ANGLE, ("C", "B", "A")).records[0].values == (12, 105, 4, 1.8)
    exact = prm.match(Kind.PROPER, ("D", "C", "B", "A"))
    assert exact.reversed and not exact.wildcard
    assert [row.values[1] for row in exact.records] == [1, 3]
    wildcard = prm.match(Kind.PROPER, ("A", "C", "B", "Z"))
    assert wildcard.wildcard and wildcard.reversed
    assert prm.match(Kind.IMPROPER, ("B", "C", "C", "A")).records[0].values[0] == 11
    assert prm.match(Kind.IMPROPER, ("A", "C", "D", "B")).wildcard
    assert prm.match(Kind.IMPROPER, ("C", "B", "C", "A")) is None
    assert prm.match(Kind.NONBONDED, ("A",)).records[0].values[-3:] == (0, -0.045, 1.5)
    assert prm.nbfix_for(("A", "B"))[0].values == (-0.15, 3, -0.1, 2.9)
    assert prm.nbfix_for(("A", "C")) == ()


def test_invalid_parameter_row_reports_line(tmp_path: Path):
    path = tmp_path / "bad.prm"
    path.write_text("BONDS\nA B NaN 1.0\n", encoding="utf-8")
    with pytest.raises(AutoCGENFFError, match="bad.prm:2: invalid bond row"):
        read_parameters(path)


def test_analogy_is_directional_and_uses_distinct_matrices_and_inner_weights():
    model = PenaltyModel(bonded={("B", "Q"): 1}, nonbonded={("A", "P"): 2})
    analog = best_analogy(Kind.ANGLE, ("A", "B", "C"), [("C", "Q", "P")], model)
    assert analog.reversed and analog.penalty == 12
    assert best_analogy(Kind.ANGLE, ("P", "Q", "C"), [("A", "B", "C")], model) is None
    assert (
        best_analogy(Kind.ANGLE, ("A", "B", "C"), [("P", "Q", "C")], model, for_charge=True) is None
    )


def test_bond_groups_prefer_preserving_ring_environment():
    model = PenaltyModel(
        nonbonded={("A", "D"): 1, ("A", "E"): 5},
        bond_groups=(BondGroup("RING", frozenset({"A", "B", "E"}), 2),),
    )
    result = best_analogy(Kind.ANGLE, ("A", "B", "C"), [("D", "B", "C"), ("E", "B", "C")], model)
    assert result.types == ("E", "B", "C") and result.penalty == 5


def test_equivalent_bond_groups_and_deterministic_tie():
    model = PenaltyModel(
        bonded={("A", "C"): 1, ("B", "D"): 1, ("B", "E"): 1},
        bond_groups=(
            BondGroup("GROUP", frozenset({"A", "B"}), 3),
            BondGroup("GROUP", frozenset({"C", "D", "E"}), 3),
        ),
    )
    candidates = [("C", "E"), ("C", "D")]
    match = best_analogy(Kind.BOND, ("A", "B"), candidates, model)
    assert match.types == ("C", "D") and match.penalty == 20
    assert match == best_analogy(Kind.BOND, ("A", "B"), reversed(candidates), model)


def test_improper_analogy_uses_declared_center_and_bonded_matrix():
    model = PenaltyModel(bonded={("B", "Z"): 2, ("D", "W"): 1})
    match = best_analogy(
        Kind.IMPROPER, ("D", "C", "A", "B"), [("Z", "A", "C", "W")], model, improper_center=3
    )
    assert match.reversed and match.penalty == 21
