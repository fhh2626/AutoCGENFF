"""Synthetic coverage for shared charge ownership and independent typing."""

from pathlib import Path

import pytest
from rdkit import Chem

from autocgenff import AutoCGENFFError, parameterize_v5
from autocgenff.molecule import Kind, inventory
from autocgenff.v5 import type_v5
from autocgenff.v5._engine import type_perception
from autocgenff.v5._perception import perceive
from autocgenff.v5._rules import parse


@pytest.mark.parametrize(
    ("smiles", "labels", "nitrogen", "impropers"),
    [
        (
            "NC(N)=[N+]=C(C)C",
            "NG2P1 CG2N1 NG2P1 NG2P1 CG2D1 CG331 CG331 " + "HGP2 " * 4 + "HGA3 " * 6,
            3,
            (1, 4),
        ),
        (
            # The non-owner cumulene carbon has two neighbors and no improper.
            "NC(N)=[N+]=C=NC",
            "NG2P1 CG2N1 NG2P1 NG2P1 CG2N2 NG2P1 CG334 " + "HGP2 " * 4 + "HGA3 " * 3,
            3,
            (1,),
        ),
        (
            "NC1=[N+]=CC2CCCN12",
            "NG2P1 CG2R53 NG2R50 CG2R52 CG3RC1 CG3C52 CG3C52 CG3C54 NG2R52 "
            "HGP2 HGP2 HGR52 HGA1 " + "HGA2 " * 6,
            2,
            (1,),
        ),
    ],
)
def test_unique_guanidinium_owns_seed_independently_of_atom_type(
    tmp_path: Path, smiles: str, labels: str, nitrogen: int, impropers: tuple[int, ...]
) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    expected_types = tuple(labels.split())
    # Exact N/A increments isolate the seeds; they are not a charge model for these ions.
    sections = {
        Kind.BOND: ("BONDS", "1 2"),
        Kind.ANGLE: ("ANGLES", "1 2"),
        Kind.PROPER: ("DIHEDRALS", "1 1 0"),
    }
    keys: dict[Kind, set[tuple[str, ...]]] = {kind: set() for kind in sections}
    for term in inventory(molecule):
        key = tuple(expected_types[i] for i in term.atoms)
        keys[term.kind].add(min(key, key[::-1]))
    rows = []
    for kind, (section, values) in sections.items():
        rows.append(section)
        rows.extend(f"{' '.join(key)} {values} !<% N/A %>!" for key in sorted(keys[kind]))
    prm = "\n".join([*rows, "END", ""])
    for name in ("par_all36_cgenff.prm", "par_all36_cgenff_increments.prm"):
        (tmp_path / name).write_text(prm, encoding="utf-8")
    (tmp_path / "top_all36_cgenff.rtf").write_text("RESI SYN 0\nEND\n", encoding="utf-8")

    order = list(range(molecule.GetNumAtoms()))
    for permutation in (order, order[::-1], order[::2] + order[1::2]):
        permuted = Chem.RenumberAtoms(molecule, permutation)
        typed = type_v5(permuted)
        assert typed.is_complete
        assert typed.types == tuple(expected_types[i] for i in permutation)
        expected_seed = [int(i == 1) for i in permutation]
        assert [a.rule_charge for a in typed.atoms] == expected_seed
        assert [a.formal_charge for a in typed.atoms] == [int(i == nitrogen) for i in permutation]
        assert [a.improper_center for a in typed.atoms] == [i in impropers for i in permutation]
        assigned = parameterize_v5(permuted, tmp_path)
        assert [a.partial_charge for a in assigned.atoms] == expected_seed
        assert {t.term.atoms[0] for t in assigned.terms if t.term.kind == Kind.IMPROPER} == {
            permutation.index(i) for i in impropers
        }
        for typed_atom, assigned_atom in zip(typed.atoms, assigned.atoms, strict=True):
            assert assigned_atom.atom_type == typed_atom.atom_type
            assert assigned_atom.formal_charge == typed_atom.formal_charge
            assert assigned_atom.rule_charge == typed_atom.rule_charge
            assert assigned_atom.improper_center == typed_atom.improper_center
            assert assigned_atom.trace == typed_atom.trace


def test_shared_seed_preserves_independent_cation_and_anion() -> None:
    typed = type_v5("NC(N)=[N+]=C(C)C[NH3+].[Cl-]")
    assert [(a.index, a.rule_charge) for a in typed.atoms if a.rule_charge] == [
        (1, 1),
        (7, 1),
        (8, -1),
    ]


@pytest.mark.parametrize("smiles", ["C=[N+]=C", "CC(N)=[N+]=C", "NC(N)=[N+]=C(N)N"])
def test_zero_or_two_guanidinium_centers_leave_overlap_fail_closed(smiles: str) -> None:
    assert sum(a.rule_charge for a in type_v5(smiles).atoms) == 2
    with pytest.raises(AutoCGENFFError, match="preserve formal charge"):
        parameterize_v5(smiles, Path("absent"))


def test_sibling_neighbor_group_reserves_the_other_nitrogen() -> None:
    program = parse("""
    cat main
    typ GUA : el C ne ( el N bo 2 fc 1 ) ( el N bo 1 ) ( el N bo 1 ) iminium_charge
    typ IMI : el C ne ( el N fc 0 ) ( el N bo 2 ) iminium_charge
    typ X   :
    end
    """)
    molecule = Chem.AddHs(Chem.MolFromSmiles("NC(N)=[N+]=C=NC"))
    order = list(range(molecule.GetNumAtoms()))
    for permutation in (order, order[::-1]):
        perception = perceive(Chem.RenumberAtoms(molecule, permutation))
        typed = type_perception(program, perception)
        assert [a.rule_charge for a in typed] == [int(i == 1) for i in permutation]
        assert sum(perception.charges) == 1


def test_second_double_bonded_cation_prevents_claim_attribution() -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("NC(N)=[N+]=C=[NH2+]"))
    order = list(range(molecule.GetNumAtoms()))
    for permutation in (order, order[::-1]):
        typed = type_v5(Chem.RenumberAtoms(molecule, permutation))
        assert [a.rule_charge for a in typed.atoms] == [int(i in (1, 4)) for i in permutation]
        assert [a.formal_charge for a in typed.atoms] == [int(i in (3, 5)) for i in permutation]


def test_branch_without_iminium_seed_stays_unchanged() -> None:
    # The right C takes CG25X, so it already emits no seed despite C=N+.
    typed = type_v5("NC(N)=[N+]=C1CCOC1")
    assert typed.types[1:5] == ("CG2N1", "NG2P1", "NG2P1", "CG25C1")
    assert [(a.index, a.rule_charge) for a in typed.atoms if a.rule_charge] == [(1, 1)]


@pytest.mark.parametrize(
    ("left_action", "right_action", "nitrogen_action", "expected"),
    [
        ("", "iminium_charge", "", (0, 0, 1)),
        ("iminium_charge", "charge 1", "", (1, 0, 1)),
        ("charge 1", "iminium_charge", "", (1, 0, 1)),
        ("iminium_charge", "iminium_charge", "", (1, 0, 0)),
        ("iminium_charge", "iminium_charge", "charge 1", (1, 1, 1)),
    ],
)
def test_resolver_requires_two_corresponding_iminium_actions(
    left_action: str, right_action: str, nitrogen_action: str, expected: tuple[int, ...]
) -> None:
    program = parse(f"""
    cat main
    typ GUA : el C ne ( el N bo 2 ) ( el N bo 1 ) ( el N bo 1 ) {left_action}
    sub IMI : el C ne ( el N bo 2 ) iminium_charge
    typ NIT : el N fc 1 {nitrogen_action}
    typ X   :
    end
    cat IMI
    typ IMI : ne ( el N bo 2 ) {right_action}
    end
    """)
    p = perceive(Chem.AddHs(Chem.MolFromSmiles("NC(N)=[N+]=C(C)C")))
    typed = type_perception(program, p)
    assert tuple(typed[i].rule_charge for i in (1, 3, 4)) == expected


@pytest.mark.parametrize(
    ("smiles", "category", "label", "seed"),
    [
        # Both bonds at N+ are endocyclic C=N, in a mixed and an all-sp2 ring respectively.
        ("NC1=[N+]=CC2CCCN12", "NG2D", "NG2R50", [0, 1, 0, 0]),
        ("NC1=[N+]=CC(=C)N1", "NG2R5", "NG2R50", [0, 1, 0, 0]),
        # Ordinary singly double-bonded imines follow the same legacy dispatch.
        ("CC1=[NH+]CCC1", "NG2D", "NG2R52", [0, 1, 0, 0]),
        ("CC1=[NH+]C=CN1", "NG2R5", "NG2R52", [0, 1, 0, 0]),
        ("CC1=NCCC1", "NG2D", "NG2R50", [0, 0, 0, 0]),
        ("CN=[N+]=[N-]", "NG", "NG1T1", [0, 0, 0, 0]),
    ],
)
def test_ring_imine_and_azide_nitrogen_dispatch(
    smiles: str, category: str, label: str, seed: list[int]
) -> None:
    typed = type_v5(smiles)
    leaf = typed.atoms[2].trace[-1]
    assert leaf.startswith(f"{category}:") and leaf.endswith(f"typ {label}")
    assert [a.rule_charge for a in typed.atoms[:4]] == seed
