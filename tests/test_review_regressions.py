"""Regression fixtures for the parser, assignment and CLI review findings."""

import json
from pathlib import Path

import pytest
from rdkit import Chem

from autocgenff import (
    AssignmentOptions,
    AutoCGENFFError,
    ChargeModel,
    PenaltyModel,
    TypingRule,
    parameterize,
    prepare_molecule,
    read_parameters,
    read_topology,
)
from autocgenff.assignment import _lookup_term
from autocgenff.charges import assign_charges
from autocgenff.cli import main
from autocgenff.fitting import TypedReference, fit_increments
from autocgenff.molecule import Kind, Term, inventory
from autocgenff.topology import match_template
from autocgenff.typing import type_atoms


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("conflicting_evidence", [False, True])
def test_improper_analogy_requires_verified_source_center(tmp_path, conflicting_evidence):
    prm = read_parameters(_write(tmp_path, "test.prm", "IMPROPER\nP Q P C 20 0 0\nEND\n"))
    # P-Q-P-C has no geometry in the PRM. Treating P as the center gives false TPS 10.
    penalty = PenaltyModel(bonded={("Z", "P"): 1})
    assert _lookup_term(prm, Kind.IMPROPER, ("Z", "Q", "P", "C"), penalty, 0)[0] == "missing"
    source = """RESI SOURCE 0
ATOM A1 P 0
ATOM A2 Q 0
ATOM A3 P 0
ATOM A4 C 0
BOND A4 A1 A4 A2 A4 A3
IMPR A1 A2 A3 A4
"""
    if conflicting_evidence:
        source += source.replace("SOURCE", "CONFLICT").replace(
            "A4 A1 A4 A2 A4 A3", "A1 A4 A1 A2 A1 A3"
        )
    rtf = read_topology(_write(tmp_path, "test.rtf", source + "END\n"))
    rules = (
        TypingRule("carbon", "[#6]", "T"),
        TypingRule("chlorine", "[#17]", "P"),
        TypingRule("bromine", "[#35]", "Q"),
        TypingRule("fluorine", "[#9]", "P"),
        TypingRule("hydrogen", "[#1]", "H"),
    )
    options = AssignmentOptions(
        rules=rules,
        impropers=((0, 1, 2, 3),),
        topology=rtf,
        penalties=PenaltyModel(bonded={("T", "C"): 1}),
    )
    result = parameterize("C(Cl)(Br)F", prm, options)
    improper = next(term for term in result.terms if term.term.kind == Kind.IMPROPER)
    if conflicting_evidence:
        assert improper.status == "missing" and improper.match is None
    else:
        assert improper.status == "analogy" and improper.penalty == 10
        assert improper.match.reversed and improper.match.records[0].types == ("P", "Q", "P", "C")


@pytest.mark.parametrize(
    ("smiles", "carbons", "hydrogens"),
    [
        ("CC=C", ["CG331", "CG2D1", "CG2D2"], ["HGA3"] * 3 + ["HGA4"] + ["HGA5"] * 2),
        (
            "CCC=C",
            ["CG331", "CG321", "CG2D1", "CG2D2"],
            ["HGA3"] * 3 + ["HGA2"] * 2 + ["HGA4"] + ["HGA5"] * 2,
        ),
        ("CC=CC", ["CG331", "CG2D1", "CG2D1", "CG331"], ["HGA3"] * 3 + ["HGA4"] * 2 + ["HGA3"] * 3),
    ],
)
def test_isolated_aliphatic_alkenes(smiles, carbons, hydrogens):
    assert [item.atom_type for item in type_atoms(prepare_molecule(smiles))] == carbons + hydrogens


@pytest.mark.parametrize("smiles", ["CC=CC=C", "CC=CCl", "OCC=C", "CC1=CCCCC1"])
def test_allylic_extension_does_not_type_unsupported_alkenes(smiles):
    with pytest.raises(AutoCGENFFError, match="No atom typing rule"):
        type_atoms(prepare_molecule(smiles))


@pytest.mark.parametrize(
    ("patterns", "order"),
    [
        (["X X A B", "X D A X", "X D A B", "C X X B"], ("C", "D", "A", "B")),
        (["X X A B", "X D A X", "B A D X"], ("B", "A", "D", "C")),
        (["X X A B", "X D A X"], ("C", "D", "A", "B")),
    ],
)
def test_improper_wildcards_have_charmm_family_priority(tmp_path, patterns, order):
    rows = "".join(f"{pattern} {i} 0 0\n" for i, pattern in enumerate(patterns, 1))
    prm = read_parameters(_write(tmp_path, "test.prm", f"IMPROPERS\n{rows}END\n"))
    for query in (order, order[::-1]):
        match = prm.match(Kind.IMPROPER, query)
        assert match.wildcard and match.records[0].values[0] == len(patterns)


@pytest.mark.parametrize("auto", ["", "AUTO ANGLES DIHE\n"])
def test_template_mapping_compares_only_effective_dependencies(tmp_path, auto):
    atoms = "ATOM C1 CT 0\nATOM C2 CT 0\n"
    for name in ("H11", "H12", "H13", "H21", "H22", "H23"):
        atoms += f"ATOM {name} HT 0\n"
    contents = (
        "MASS 1 CT 12 C\nMASS 2 HT 1 H\n"
        + auto
        + "RESI ETH 0\n"
        + atoms
        + "BOND C1 C2 C1 H11 C1 H12 C1 H13 C2 H21 C2 H22 C2 H23\n"
        + "ANGL H11 C1 C2\nDIHE H11 C1 C2 H21\nEND\n"
    )
    rtf = read_topology(_write(tmp_path, "test.rtf", contents))
    mol = prepare_molecule("CC")
    if not auto:
        with pytest.raises(AutoCGENFFError, match="Ambiguous mapping"):
            match_template(mol, rtf, "ETH")
        return
    template = match_template(mol, rtf, "ETH")
    assert not template.angles and not template.propers
    terms = inventory(mol, auto_angles=template.auto_angles, auto_propers=template.auto_propers)
    assert sum(term.kind == Kind.ANGLE for term in terms) == 12
    assert sum(term.kind == Kind.PROPER for term in terms) == 9


@pytest.mark.parametrize("source", ["A B C D", "X B C X"])
def test_fourier_matches_report_orientation_per_native_row(tmp_path, source):
    reverse = " ".join(reversed(source.split()))
    prm = read_parameters(
        _write(tmp_path, "test.prm", f"DIHE\n{source} 1 1 10\n{reverse} 2 2 20\n")
    )
    for query, orientations in ((tuple("ABCD"), (False, True)), (tuple("DCBA"), (True, False))):
        match = prm.match(Kind.PROPER, query)
        assert match.record_reversed == orientations and match.reversed is None
        assert [row.types for row in match.records] == [
            tuple(source.split()),
            tuple(reverse.split()),
        ]
        assert [row.values[2] for row in match.records] == [10, 20]
    if source.startswith("A"):
        penalties = PenaltyModel(nonbonded={("Z", "D"): 1})
        status, _, match = _lookup_term(prm, Kind.PROPER, tuple("ZCBA"), penalties, None)
        assert status == "analogy" and match.record_reversed == (True, False)


def test_zero_torsion_cutoff_is_invalid():
    with pytest.raises(AutoCGENFFError, match="positive"):
        ChargeModel({("A", "Q", "C", "D"): (0.1, 0.2, 0.3)}, torsion_cutoff=0)


def test_rejects_multiline_smiles_but_allows_blank_lines():
    with pytest.raises(AutoCGENFFError, match="one.*SMILES"):
        prepare_molecule("CCO\n\nCC")
    assert prepare_molecule("\nCCO\r\n\t").GetNumAtoms() == 9


_WATER = """MASS 1 OW 16 O
MASS 2 HW 1 H
RESI WATER 0
ATOM O1 OW 0
ATOM H1 HW 0
ATOM H2 HW 0
BOND O1 H1 O1 H2
"""
_WATER_PRM = "BONDS\nOW HW 200 1\nNONBONDED\nOW 0 -0.1 1.8\nHW 0 -0.01 0.1\nEND\n"


@pytest.mark.parametrize(
    ("defaults", "override", "remaining"),
    [
        ("DEFA FIRS NONE LAST CAP\n", "PATCHING FIRS NONE\n", ("CAP",)),
        ("DEFA FIRS NTER LAST CTER\n", "PATCHING FIRST GLYP\n", ("GLYP", "CTER")),
        ("DEFA FIRS ACE LAST NONE\n", "PATCHING LAST NONE\n", ("ACE",)),
        ("DEFA FIRS ACE LAST CAP\n", "PATCHING FIRS NONE\nPATCHING LAST NONE\n", ()),
    ],
)
def test_partial_patching_preserves_the_other_terminus(tmp_path, defaults, override, remaining):
    rtf = read_topology(_write(tmp_path, "test.rtf", defaults + _WATER + override + "END\n"))
    assert rtf.residues["WATER"].patches == remaining
    if remaining:
        with pytest.raises(AutoCGENFFError, match="unsupported topology"):
            match_template(prepare_molecule("O"), rtf, "WATER")
    else:
        assert match_template(prepare_molecule("O"), rtf, "WATER") is not None


@pytest.mark.parametrize("use_group_in_query", [False, True])
def test_named_identity_rejects_enhanced_stereo(tmp_path, use_group_in_query):
    plain = "F[C@H](Cl)Br"
    group = plain + " |o1:1|"
    rtf = read_topology(
        _write(
            tmp_path,
            "test.rtf",
            """MASS 1 CT 12 C
MASS 2 HT 1 H
MASS 3 FT 19 F
MASS 4 CLT 35 Cl
MASS 5 BRT 80 Br
RESI HAL 0
ATOM C CT 0
ATOM H HT 0
ATOM F FT 0
ATOM CL CLT 0
ATOM BR BRT 0
BOND C H C F C CL C BR
END
""",
        )
    )
    assert match_template(
        prepare_molecule(plain), rtf, "HAL", reference_smiles=plain
    ).verified_reference_smiles
    if use_group_in_query:
        query = prepare_molecule(Chem.MolFromSmiles(group))
        assert query.GetStereoGroups()
        with pytest.raises(AutoCGENFFError, match="Enhanced stereochemistry"):
            match_template(query, rtf, "HAL", reference_smiles=plain)
    else:
        with pytest.raises(AutoCGENFFError, match="CXSMILES"):
            match_template(prepare_molecule(plain), rtf, "HAL", reference_smiles=group)


@pytest.mark.parametrize("index", [False, 0.0])
def test_python_atom_map_requires_integer_indices(tmp_path, index):
    rtf = read_topology(_write(tmp_path, "test.rtf", _WATER + "END\n"))
    with pytest.raises(AutoCGENFFError, match="integer atom index"):
        match_template(
            prepare_molecule("O"), rtf, "WATER", atom_map={"O1": index, "H1": 1, "H2": 2}
        )


@pytest.mark.parametrize("duplicate", ["O1", "o1"])
def test_cli_atom_map_rejects_duplicate_names(tmp_path, capfd, duplicate):
    prm = _write(tmp_path, "test.prm", _WATER_PRM)
    rtf = _write(tmp_path, "test.rtf", _WATER + "END\n")
    atom_map = _write(tmp_path, "map.json", f'{{"{duplicate}":99,"O1":0,"H1":1,"H2":2}}')
    args = ["--prm", str(prm), "--rtf", str(rtf), "--template", "WATER", "--smiles", "O"]
    assert main([*args, "--atom-map", str(atom_map)]) == 2
    capture = capfd.readouterr()
    assert not capture.out and "Duplicate" in json.loads(capture.err)["error"]


@pytest.mark.parametrize(
    "header", ["THETA", "NBOND", "NONBONDED nbxmod 5\t-\ncdiel vatom vdistance cutnb 14"]
)
def test_prm_accepts_free_field_continuation_and_section_aliases(tmp_path, header):
    kind, types, values = (
        (Kind.ANGLE, ("A", "B", "A"), "10 180")
        if header == "THETA"
        else (Kind.NONBONDED, ("A",), "0 -0.1 1.8")
    )
    row = " ".join((*types, values))
    prm = read_parameters(
        _write(tmp_path, "test.prm", f"BOND\nA B 100 1.5\n{header}\n{row}\nEND\n")
    )
    assert prm.match(kind, types).records[0].values == tuple(map(float, values.split()))


@pytest.mark.parametrize("case", ["arguments", "smiles", "baseline", "smarts"])
def test_cli_input_errors_are_a_single_stderr_json_document(tmp_path, capfd, case):
    prm = _write(tmp_path, "test.prm", _WATER_PRM)
    args = ["--prm", str(prm)]
    if case == "smiles":
        args += ["--smiles", "C("]
    elif case == "baseline":
        baseline = _write(tmp_path, "formal.json", "[" + "1" + "0" * 400 + "]")
        args += ["--smiles", "O", "--formal-baseline", str(baseline)]
    elif case == "smarts":
        rules = _write(tmp_path, "rules.json", '[{"name":"broken","smarts":"[", "atom_type":"A"}]')
        args += ["--smiles", "O", "--rules", str(rules)]
    assert main(args) == 2
    capture = capfd.readouterr()
    assert not capture.out and json.loads(capture.err)["error"]


def test_protein_rtf_preserves_gly_last_patch():
    path = Path(__file__).resolve().parents[1] / "force_fields" / "top_all36_prot.rtf"
    if not path.is_file():
        pytest.skip("User-supplied protein reference RTF not present")
    assert read_topology(path).residues["GLY"].patches == ("GLYP", "CTER")


def test_named_identity_rejects_atropisomeric_query_or_reference(tmp_path):
    plain = "CC1=CC=CC(I)=C1N1C(C)=CC=C1Br"
    up, down = plain + " |wU:7.7|", plain + " |wD:7.7|"
    if not Chem.MolFromSmiles(up).GetBondWithIdx(7).GetStereo().name.startswith("STEREOATROP"):
        pytest.skip("This RDKit version does not parse CXSMILES atropisomerism")
    mol, other = (prepare_molecule(Chem.MolFromSmiles(text)) for text in (up, down))
    stereo = (mol.GetBondWithIdx(7).GetStereo().name, other.GetBondWithIdx(7).GetStereo().name)
    assert set(stereo) == {"STEREOATROPCCW", "STEREOATROPCW"}
    assert not mol.GetStereoGroups() and not other.GetStereoGroups()

    elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
    rows = [f"MASS {i} {element} 1 {element}" for i, element in enumerate(elements, 1)]
    rows.append("RESI AXIAL 0")
    rows.extend(f"ATOM A{atom.GetIdx()} {atom.GetSymbol()} 0" for atom in mol.GetAtoms())
    rows.extend(
        f"BOND A{bond.GetBeginAtomIdx()} A{bond.GetEndAtomIdx()}" for bond in mol.GetBonds()
    )
    rtf = read_topology(_write(tmp_path, "axial.rtf", "\n".join([*rows, "END", ""])))
    atom_map = {f"A{i}": i for i in range(mol.GetNumAtoms())}
    plain_mol = prepare_molecule(plain)
    assert match_template(
        plain_mol, rtf, "AXIAL", atom_map=atom_map, reference_smiles=plain
    ).verified_reference_smiles
    for query, reference in ((mol, down), (mol, plain), (other, up), (mol, None)):
        with pytest.raises(AutoCGENFFError, match="Atropisomeric stereochemistry"):
            match_template(query, rtf, "AXIAL", atom_map=atom_map, reference_smiles=reference)
    with pytest.raises(AutoCGENFFError, match="CXSMILES"):
        match_template(plain_mol, rtf, "AXIAL", atom_map=atom_map, reference_smiles=up)


@pytest.mark.parametrize("reverse_second", [False, True])
def test_exact_improper_matches_retain_both_native_rows(tmp_path, reverse_second):
    forward = ("A", "B", "C", "D")
    second = forward[::-1] if reverse_second else forward
    prm = read_parameters(
        _write(tmp_path, "test.prm", f"IMPROPER\nA B C D 2 1 180\n{' '.join(second)} 3 2 0\nEND\n")
    )
    assert len(tuple(prm.iter_groups(Kind.IMPROPER))) == 1
    for query, reverse in ((forward, False), (forward[::-1], True)):
        match = prm.match(Kind.IMPROPER, query)
        assert not match.wildcard
        assert [row.types for row in match.records] == [forward, second]
        assert [row.values for row in match.records] == [(2, 1, 180), (3, 2, 0)]
        assert [row.line for row in match.records] == [2, 3]
        assert match.record_reversed == (reverse, reverse != reverse_second)
        assert match.reversed is (None if reverse_second else reverse)


@pytest.mark.parametrize(("section", "kind"), [("DIHE", "proper"), ("IMPROPER", "improper")])
@pytest.mark.parametrize("pattern", ["A X C D", "D C X A"])
def test_unsupported_torsion_wildcards_are_rejected(tmp_path, section, kind, pattern):
    path = _write(tmp_path, "test.prm", f"{section}\n{pattern} 2 1 180\nEND\n")
    with pytest.raises(AutoCGENFFError, match=f"test.prm:2: invalid {kind} row:.*wildcard"):
        read_parameters(path)


@pytest.mark.parametrize("charges", [(0, 0), (-1, 1)])
def test_integer_target_charges_work_in_staged_fitting(charges):
    mol = prepare_molecule("ClCl")
    types = ("LEFT", "RIGHT")
    result = fit_increments([TypedReference(mol, types, charges)], max_order=2, ridge=0)
    assert result.model.increments[types] == pytest.approx((charges[1],))
    assert result.stages[0].columns == 1
    assert result.stages[0].rms_error == pytest.approx(0)


@pytest.mark.parametrize("index", [False, 0.0])
@pytest.mark.parametrize(
    ("argument", "path", "kind", "error"),
    [
        ("explicit_angles", (0, 1, 2), Kind.ANGLE, "Invalid declared covalent path"),
        ("explicit_propers", (0, 1, 2, 8), Kind.PROPER, "Invalid declared covalent path"),
        ("impropers", (0, 1, 3, 4), Kind.IMPROPER, "Invalid declared improper"),
    ],
)
def test_declared_covalent_paths_require_integer_indices(index, argument, path, kind, error):
    mol = prepare_molecule("CCO")
    options = {"auto_angles": False, "auto_propers": False}
    assert any(
        term.kind == kind and term.atoms == path
        for term in inventory(mol, **options, **{argument: (path,)})
    )
    with pytest.raises(AutoCGENFFError, match=error):
        inventory(mol, **options, **{argument: ((index, *path[1:]),)})


@pytest.mark.parametrize("index", [False, 0.0])
def test_python_assignment_rejects_non_integer_improper_indices(tmp_path, index):
    prm = read_parameters(_write(tmp_path, "test.prm", _WATER_PRM))
    result = parameterize("CCO", prm, AssignmentOptions(impropers=((0, 1, 3, 4),)))
    assert [term.term.atoms for term in result.terms if term.term.kind == Kind.IMPROPER] == [
        (0, 1, 3, 4)
    ]
    with pytest.raises(AutoCGENFFError, match="Invalid declared improper"):
        parameterize("CCO", prm, AssignmentOptions(impropers=((index, 1, 3, 4),)))


def test_prm_accepts_standalone_nbon(tmp_path):
    prm = read_parameters(_write(tmp_path, "test.prm", "NBON\nOW 0 -0.1 1.8\nEND\n"))
    assert prm.match(Kind.NONBONDED, ("OW",)).records[0].values == (0, -0.1, 1.8)


@pytest.mark.parametrize(("angle", "improper"), [("THET", "IMPH"), ("THETA", "IMPHI")])
def test_rtf_accepts_charmm_angle_and_improper_aliases(tmp_path, angle, improper):
    rtf = read_topology(
        _write(
            tmp_path,
            "test.rtf",
            f"""MASS 1 CT 12 C
MASS 2 HT 1 H
RESI METH 0
ATOM C CT 0
ATOM H1 HT 0
ATOM H2 HT 0
ATOM H3 HT 0
ATOM H4 HT 0
BOND C H1 C H2 C H3 C H4
{angle} H1 C H2
{improper} C H1 H2 H3
END
""",
        )
    )
    atom_map = {name: index for index, name in enumerate(("C", "H1", "H2", "H3", "H4"))}
    match = match_template(prepare_molecule("C"), rtf, "METH", atom_map=atom_map)
    assert match.angles == ((1, 0, 2),) and match.impropers == ((0, 1, 2, 3),)


@pytest.mark.parametrize("optional", [(), (-0.1,), (-0.1, 3.0)])
def test_nbfix_preserves_optional_14_columns_as_given(tmp_path, optional):
    values = (-0.2, 3.5, *optional)
    row = " ".join(map(str, values))
    prm = read_parameters(_write(tmp_path, "test.prm", f"NBFIX\nOW HW {row}\nEND\n"))
    assert prm.nbfix_for(("OW", "HW"))[0].values == values


def test_replaced_torsion_component_keeps_surviving_rows_in_source_order(tmp_path):
    prm = read_parameters(
        _write(tmp_path, "test.prm", "DIHE\nA B C D 1 1 0\nA B C D 2 2 180\nD C B A 3 1 90\nEND\n")
    )
    match = prm.match(Kind.PROPER, ("A", "B", "C", "D"))
    assert [row.line for row in match.records] == [3, 4]
    assert [row.values for row in match.records] == [(2, 2, 180), (3, 1, 90)]
    assert match.record_reversed == (False, True)


_CL2_RTF = "MASS 1 CL 35 Cl\nRESI CL2 0\nATOM A CL 0\nATOM B CL 0\nBOND A B\nEND\n"
_CL2_PRM = "BONDS\nCL CL 90 2\nNONBONDED\nCL 0 -.1 2\n"


def _template_for(tmp_path, molecule, *directives, group=False):
    elements = sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})
    rows = [f"MASS {i} {element} 1 {element}" for i, element in enumerate(elements, 1)]
    rows.append("RESI TEST 0")
    if group:
        rows.append("GROU")
    rows.extend(f"ATOM A{atom.GetIdx()} {atom.GetSymbol()} 0" for atom in molecule.GetAtoms())
    kinds = {Chem.BondType.SINGLE: "BOND", Chem.BondType.DOUBLE: "DOUBLE"}
    rows.extend(
        f"{kinds[bond.GetBondType()]} A{bond.GetBeginAtomIdx()} A{bond.GetEndAtomIdx()}"
        for bond in molecule.GetBonds()
    )
    rows.extend(directives)
    topology = read_topology(_write(tmp_path, "generated.rtf", "\n".join([*rows, "END", ""])))
    return topology, {f"A{i}": i for i in range(molecule.GetNumAtoms())}


def test_named_identity_rejects_single_enum_query_bond(tmp_path):
    mol = Chem.RWMol(Chem.MolFromSmiles("ClCl"))
    mol.ReplaceBond(0, Chem.BondFromSmarts("-;@"))
    assert mol.GetBondWithIdx(0).HasQuery()
    assert mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.SINGLE
    rtf = read_topology(_write(tmp_path, "test.rtf", _CL2_RTF))
    prm = read_parameters(_write(tmp_path, "test.prm", _CL2_PRM + "END\n"))
    options = AssignmentOptions(topology=rtf, template="CL2", template_smiles="ClCl")
    assert parameterize("ClCl", prm, options).template_source.verified_reference_smiles
    with pytest.raises(AutoCGENFFError, match="Query bonds"):
        parameterize(mol, prm, options)
    with pytest.raises(AutoCGENFFError, match="Query bonds"):
        match_template(mol, rtf, "CL2", reference_smiles="ClCl")


@pytest.mark.parametrize("smiles", ["F/C=C/F", r"F/C=C\F"])
def test_named_identity_preserves_double_bond_stereo_without_directions(tmp_path, smiles):
    source = Chem.MolFromSmiles(smiles)
    for bond in source.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
    assert Chem.MolToSmiles(source) == smiles
    prepared = prepare_molecule(source)
    assert prepared.GetBondWithIdx(1).GetStereo() == source.GetBondWithIdx(1).GetStereo()
    assert all(b.GetBondDir() == Chem.BondDir.NONE for b in source.GetBonds())

    rtf, atom_map = _template_for(tmp_path, prepared)
    assert match_template(
        prepared, rtf, "TEST", atom_map=atom_map, reference_smiles=smiles
    ).verified_reference_smiles
    with pytest.raises(AutoCGENFFError, match="differs from the supplied template SMILES"):
        match_template(prepared, rtf, "TEST", atom_map=atom_map, reference_smiles="FC=CF")


def test_named_identity_rejects_allene_annotations_before_cleanup(tmp_path):
    plain = "FC=C=CCl"
    left, right = "FC=[C@AL1]=CCl", "FC=[C@AL2]=CCl"
    molecule = prepare_molecule(plain)
    rtf, atom_map = _template_for(tmp_path, molecule)
    assert match_template(
        molecule, rtf, "TEST", atom_map=atom_map, reference_smiles=plain
    ).verified_reference_smiles
    raw = Chem.MolFromSmiles(left, sanitize=False)
    assert raw.GetAtomWithIdx(2).GetChiralTag() == Chem.ChiralType.CHI_ALLENE
    for source in (left, right, raw):
        with pytest.raises(AutoCGENFFError, match="allene stereochemistry"):
            prepare_molecule(source)
    for query, reference in ((raw, right), (raw, None), (molecule, right)):
        with pytest.raises(AutoCGENFFError, match="allene stereochemistry"):
            match_template(query, rtf, "TEST", atom_map=atom_map, reference_smiles=reference)


@pytest.mark.parametrize("tail", ["", "\n", " \t\r\n\n", "bad record\n", "\n$$$$\n", "duplicate"])
def test_cli_sdf_allows_only_whitespace_after_one_record(tmp_path, capfd, tail):
    record = Chem.MolToMolBlock(prepare_molecule("ClCl")) + "$$$$\n"
    sdf = _write(tmp_path, "test.sdf", record + (record if tail == "duplicate" else tail))
    prm = _write(tmp_path, "test.prm", _CL2_PRM + "END\n")
    rtf = _write(tmp_path, "test.rtf", _CL2_RTF)
    code = main(["--prm", str(prm), "--rtf", str(rtf), "--template", "CL2", "--input", str(sdf)])
    out, err = capfd.readouterr()
    if tail.strip():
        assert code == 2 and not out and "Expected one SDF record" in json.loads(err)["error"]
    else:
        assert code == 0 and not err and json.loads(out)["is_complete"]


@pytest.mark.parametrize("title", ["$$$$ ethanol", "$$$$", "M  END"])
def test_sdf_boundary_markers_in_title_and_valid_multiline_data(tmp_path, title):
    source = Chem.MolFromSmiles("CCO")
    source.SetProp("_Name", title)
    source.SetProp("example", "first line\nsecond line")
    path = tmp_path / "test.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(source)
    prepared = prepare_molecule(path)
    assert prepared.GetProp("_Name") == title
    assert prepared.GetProp("example") == "first line\nsecond line"
    assert Chem.MolToSmiles(Chem.RemoveHs(prepared)) == "CCO"


@pytest.mark.parametrize("tail", ["garbage", "molblock", "molblock_with_separator"])
def test_sdf_does_not_discard_content_after_first_mol_block(tmp_path, tail):
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("O"))
    second = Chem.MolToMolBlock(Chem.MolFromSmiles("ClCl"))
    suffix = {
        "garbage": "unrecognized tail\n",
        "molblock": second,
        "molblock_with_separator": second + "$$$$\n",
    }[tail]
    assert prepare_molecule(_write(tmp_path, "valid.sdf", block)).GetNumAtoms() == 3
    with pytest.raises(AutoCGENFFError, match="one SDF record"):
        prepare_molecule(_write(tmp_path, "malformed.sdf", block + suffix))


@pytest.mark.parametrize(
    "title",
    [
        "chlorine as @<TRIPOS>MOLECULE",
        "  @<TRIPOS>MOLECULE",
        "@<tripos>molecule",
        "@<TRIPOS>MOLECULE",
    ],
)
def test_mol2_boundary_marker_in_title_is_not_a_record(tmp_path, title):
    block = f"""@<TRIPOS>MOLECULE
{title}
2 1 0 0 0
SMALL
NO_CHARGES
@<TRIPOS>ATOM
1 Cl1 0.0 0.0 0.0 Cl
2 Cl2 2.0 0.0 0.0 Cl
@<TRIPOS>BOND
1 1 2 1
"""
    path = _write(tmp_path, "one.mol2", block)
    assert Chem.MolToSmiles(prepare_molecule(path)) == "ClCl"
    with pytest.raises(AutoCGENFFError, match="one MOL2 molecule"):
        prepare_molecule(_write(tmp_path, "two.mol2", block + block))


@pytest.mark.parametrize("counts", ["1 1 0 0 0", "2 0 0 0 0"])
def test_mol2_rejects_rows_beyond_declared_counts(tmp_path, counts):
    block = f"""@<TRIPOS>MOLECULE
chlorine
{counts}
SMALL
NO_CHARGES
@<TRIPOS>ATOM
1 Cl1 0.0 0.0 0.0 Cl
2 Cl2 2.0 0.0 0.0 Cl
@<TRIPOS>BOND
1 1 2 1
"""
    with pytest.raises(AutoCGENFFError, match="MOL2.*counts"):
        prepare_molecule(_write(tmp_path, "extra.mol2", block))


@pytest.mark.parametrize("header", [">", "> trash", "> <not closed", "> <A> >", "> Reference DT13"])
def test_sdf_rejects_malformed_data_header(tmp_path, header):
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("O"))
    with pytest.raises(AutoCGENFFError, match="one SDF record"):
        prepare_molecule(_write(tmp_path, "bad.sdf", block + f"{header}\nvalue\n\n$$$$\n"))


@pytest.mark.parametrize(
    "smiles",
    ["CCO |$Ethanol$|", "CCO |$Ethanol$| .[Na+]", "CCO |atomProp:garbage|", "F[C@H](Cl)Br |o1:99|"],
)
def test_cxsmiles_is_never_silently_simplified(smiles):
    with pytest.raises(AutoCGENFFError, match="CXSMILES"):
        prepare_molecule(smiles)


def test_mol2_rejects_shadow_bond_section(tmp_path):
    block = """@<TRIPOS>MOLECULE
chlorine
3 1 0 0 0
SMALL
NO_CHARGES
@<TRIPOS>ATOM
1 Cl1 0 0 0 Cl
2 Cl2 2 0 0 Cl
3 Cl3 4 0 0 Cl
@<TRIPOS>BOND
1 1 2 1
@<TRIPOS>BOND bogus
1 1 3 1
"""
    with pytest.raises(AutoCGENFFError, match="Invalid MOL2 section marker"):
        prepare_molecule(_write(tmp_path, "shadow.mol2", block))


@pytest.mark.parametrize("keyword", ["IC", "BILD", "BUILD"])
def test_rtf_internal_coordinates_are_not_bonded_dependencies(tmp_path, keyword):
    rtf = read_topology(
        _write(tmp_path, "test.rtf", _WATER + f"{keyword} BLNK H1 O1 H2 0 0 0 104.5 .95\nEND\n")
    )
    prm = read_parameters(_write(tmp_path, "test.prm", _WATER_PRM))
    result = parameterize("O", prm, AssignmentOptions(topology=rtf, template="WATER"))
    assert result.is_complete


def test_rtf_defaults_inherit_required_patch(tmp_path):
    contents = "DEFAULTS FIRST REQ LAST NONE\n" + _CL2_RTF.replace(
        "END\n", "PRES REQ 0\nDELETE BOND A B\nEND\n"
    )
    rtf = read_topology(_write(tmp_path, "test.rtf", contents))
    assert rtf.residues["CL2"].patches == ("REQ",)
    prm = read_parameters(_write(tmp_path, "test.prm", _CL2_PRM))
    with pytest.raises(AutoCGENFFError, match="requires unsupported topology: REQ"):
        parameterize("ClCl", prm, AssignmentOptions(topology=rtf, template="CL2"))


def test_rtf_full_residue_keyword(tmp_path):
    contents = _CL2_RTF.replace("RESI CL2", "RESIDUE CL2").replace(
        "END\n", "DECLARE +N\nPRESIDUE OTHER 0\nDELETE BOND A B\nEND\n"
    )
    rtf = read_topology(_write(tmp_path, "test.rtf", contents))
    assert match_template(prepare_molecule("ClCl"), rtf, "CL2").name == "CL2"


def test_continued_rtf_internal_coordinates_do_not_swallow_patch(tmp_path):
    contents = _CL2_RTF.replace(
        "END\n", "IC A B -\n   *A B 0 0 0 0 0\nPATC FIRS REQ LAST NONE\nEND\n"
    )
    rtf = read_topology(_write(tmp_path, "test.rtf", contents))
    with pytest.raises(AutoCGENFFError, match="requires unsupported topology: REQ"):
        match_template(prepare_molecule("ClCl"), rtf, "CL2")


@pytest.mark.parametrize("token", ["DIHE", "PHI"])
def test_rtf_proper_alias_and_group_divider(tmp_path, token):
    mol = prepare_molecule("CCCC")
    rtf, atom_map = _template_for(tmp_path, mol, f"{token} A0 A1 A2 A3", group=True)
    match = match_template(mol, rtf, "TEST", atom_map=atom_map)
    assert match.propers == ((0, 1, 2, 3),) and not match.auto_propers


def test_display_control_is_ignored_in_prm_sections_and_rtf_templates(tmp_path):
    rtf_data = _WATER.replace("ATOM O1", "PRINT ON\nATOM O1").replace("BOND", "PRIN OFF\nBOND")
    prm_data = _WATER_PRM.replace("OW HW 200", "PRINT ON\nOW HW 200").replace(
        "NONBONDED\n", "NONBONDED\nPRINT OFF\n"
    )
    rtf = read_topology(_write(tmp_path, "test.rtf", rtf_data + "END\n"))
    prm = read_parameters(_write(tmp_path, "test.prm", prm_data))
    result = parameterize("O", prm, AssignmentOptions(topology=rtf, template="WATER"))
    assert result.is_complete
    bonds = [term for term in result.terms if term.term.kind == Kind.BOND]
    assert [term.match.records[0].line for term in bonds] == [3, 3]


@pytest.mark.parametrize("keyword", ["PRINT", "PRIN"])
def test_bare_prm_display_control_is_ignored(tmp_path, keyword):
    contents = f"BOND\nA B 10 1\n{keyword}\nC D 20 1\nEND\n"
    prm = read_parameters(_write(tmp_path, "test.prm", contents))
    assert prm.match(Kind.BOND, ("A", "B")).records[0].line == 2
    assert prm.match(Kind.BOND, ("C", "D")).records[0].line == 4


@pytest.mark.parametrize(
    ("header", "pattern", "positive", "negative"),
    [
        ("NONB", "C*", ["C", "CL"], "XC"),
        ("NONBONDED", "*L", ["L", "CL"], "CL1"),
        ("NONB", "C%", ["CL"], "CL1"),
        ("NONB", "CL#", ["CL", "CL12"], "CLA"),
        ("NONB", "CL+", ["CL1"], "CL12"),
    ],
)
def test_native_nonbonded_wildcard_rows(tmp_path, header, pattern, positive, negative):
    contents = f"{header}\n* ordinary comment\n{pattern} 0 -.2 1.3\nOW 0 -.1 2\nEND\n"
    prm = read_parameters(_write(tmp_path, "test.prm", contents))
    for atom_type in positive:
        match = prm.match(Kind.NONBONDED, (atom_type,))
        assert match.wildcard and match.record_reversed == (False,)
        assert (match.records[0].types, match.records[0].values, match.records[0].line) == (
            (pattern,),
            (0, -0.2, 1.3),
            3,
        )
    assert prm.match(Kind.NONBONDED, (negative,)) is None


@pytest.mark.parametrize("row", ["*L 0 -.1", "*L 0 -.typo 3"])
def test_malformed_leading_star_prm_pattern_is_not_a_comment(tmp_path, row):
    with pytest.raises(AutoCGENFFError, match=r"test\.prm:3:.*invalid nonbonded row"):
        read_parameters(_write(tmp_path, "test.prm", f"NONB\nC* 0 -.1 2\n{row}\nEND\n"))


def test_bare_star_nbfix_distinguishes_numeric_fields_from_a_comment(tmp_path):
    text = "NBFIX\n* GENERATED JULY 2026\n* Copyright 2026 AutoCGENFF\n* CL -.1D0 2\n"
    prm = read_parameters(_write(tmp_path, "test.prm", text + "END\n"))
    assert prm.match(Kind.NBFIX, ("CL", "A")).records[0].line == 4
    with pytest.raises(AutoCGENFFError, match=r"test\.prm:5:.*invalid nbfix row"):
        read_parameters(_write(tmp_path, "test.prm", text + "* CL NaN Infinity\nEND\n"))


def test_replaced_improper_wildcard_does_not_move_ahead_of_earlier_row(tmp_path):
    text = "IMPROPER\nX B C D 10 0 0\nA B C X 20 0 0\nX B C D 30 0 0\nEND\n"
    prm = read_parameters(_write(tmp_path, "test.prm", text))
    assert prm.match(Kind.IMPROPER, ("A", "B", "C", "D")).records[0].line == 3
    assert [group[0].line for group in prm.iter_groups(Kind.IMPROPER)] == [3, 4]


def test_nonbonded_patterns_are_native_assignment_with_source_priority(tmp_path):
    contents = "BOND\nCL CL 90 2\nNONB\nC* 0 -.1 2\nCL 0 -.2 3\n*L -\n0 -.3 4\nEND\n"
    path = _write(tmp_path, "test.prm", contents)
    prm = read_parameters(path)
    exact = prm.match(Kind.NONBONDED, ("CL",))
    assert not exact.wildcard and exact.records[0].line == 5
    assert prm.match(Kind.NONBONDED, ("CCL",)).records[0].line == 6
    prm = read_parameters(_write(tmp_path, "test.prm", contents.replace("CL 0 -.2 3\n", "")))
    rtf = read_topology(_write(tmp_path, "test.rtf", _CL2_RTF))
    result = parameterize("ClCl", prm, AssignmentOptions(topology=rtf, template="CL2"))
    assert result.is_complete
    lj = [term for term in result.terms if term.term.kind == Kind.NONBONDED]
    assert len(lj) == 2 and all(term.status == "wildcard" and term.penalty == 0 for term in lj)


@pytest.mark.parametrize(
    ("row", "types", "line"),
    [
        ("CL -\n* -.2 2", ("CL", "*"), 2),
        ("* -\nCL -.2 2", ("*", "CL"), 2),
        ("CL -\n* ordinary comment\n*L -\n-.2 2", ("CL", "*L"), 2),
    ],
)
def test_nbfix_asterisk_row_uses_continuation_context(tmp_path, row, types, line):
    prm = read_parameters(_write(tmp_path, "test.prm", f"NBFIX\n{row}\nCL CL -.3 3\nEND\n"))
    match = prm.match(Kind.NBFIX, ("CL", "AL"))
    assert match.wildcard
    record = match.records[0]
    assert (record.types, record.line, record.values) == (types, line, (-0.2, 2))
    assert prm.match(Kind.NBFIX, ("CL", "CL")).records[0].values == (-0.3, 3)


@pytest.mark.parametrize("pattern", ["C*", "*L", "*", "C%", "CL#", "X"])
def test_named_assignment_includes_applicable_native_nbfix_wildcard(tmp_path, pattern):
    contents = _CL2_PRM + f"NBFIX\n{pattern} CL -.2 3\n{pattern} NA -.3 4\nEND\n"
    rtf = read_topology(_write(tmp_path, "test.rtf", _CL2_RTF))
    prm = read_parameters(_write(tmp_path, "test.prm", contents))
    result = parameterize("ClCl", prm, AssignmentOptions(topology=rtf, template="CL2"))
    assert result.is_complete
    match = prm.match(Kind.NBFIX, ("CL", "CL"))
    assert match.wildcard and match.record_reversed == (False,)
    assert result.nbfix == match.records
    assert [(row.types, row.values, row.line) for row in result.nbfix] == [
        ((pattern, "CL"), (-0.2, 3.0), 6)
    ]


def test_nbfix_inventory_and_lookup_preserve_charmm_source_priority(tmp_path):
    contents = """* PRM title
NBFIX
* ordinary comment
X X -.1 2
CL A12 -.2 3
A+# CL -.3 4
A12 CL -.4 5
A1% CL -.5 6
END
"""
    prm = read_parameters(_write(tmp_path, "test.prm", contents))
    assert [row.line for row in prm.nbfix_for(("CL", "A12"))] == [4, 6, 7, 8]
    match = prm.match(Kind.NBFIX, ("CL", "A12"))
    assert match.wildcard and match.reversed and match.records[0].line == 8
    # '+' requires one digit; '#' permits trailing digits, including none.
    assert prm.match(Kind.NBFIX, ("A1", "CL")).records[0].line == 6
    assert prm.match(Kind.NBFIX, ("A", "CL")).records[0].line == 4
    assert prm.nbfix_for(()) == ()


@pytest.mark.parametrize("section", ["DIHE", "IMPR", "IMPROPER"])
def test_prm_rejects_fractional_torsion_multiplicity(tmp_path, section):
    path = _write(tmp_path, "test.prm", f"{section}\nA B C D 10 1.5 180\nEND\n")
    with pytest.raises(AutoCGENFFError, match=r"test\.prm:2:.*invalid torsion multiplicity"):
        read_parameters(path)


def test_full_explicit_atom_map_validates_without_recursive_search(tmp_path):
    molecule = prepare_molecule("C" * 335)
    assert molecule.GetNumAtoms() == 1007
    rtf, atom_map = _template_for(tmp_path, molecule)
    match = match_template(molecule, rtf, "TEST", atom_map=atom_map)
    assert [atom.name for atom in match.atoms] == list(atom_map)
    # Two interior carbon atoms have equal degrees and elements, but not these edges.
    atom_map["A1"], atom_map["A4"] = atom_map["A4"], atom_map["A1"]
    with pytest.raises(AutoCGENFFError, match="connectivity and bond orders"):
        match_template(molecule, rtf, "TEST", atom_map=atom_map)


@pytest.mark.parametrize(
    ("key", "value", "smarts"),
    [("target", True, "[#6]-[#8]"), ("target", False, "[#8]"), ("priority", True, "[#8]")],
)
def test_cli_rejects_boolean_smarts_indices_and_priorities(tmp_path, capfd, key, value, smarts):
    rule = {"name": "probe", "smarts": smarts, "atom_type": "OW", key: value}
    with pytest.raises(AutoCGENFFError, match="priority and target must be integers"):
        TypingRule(**rule)
    rules = _write(tmp_path, "rules.json", json.dumps([rule]))
    prm = _write(tmp_path, "test.prm", _WATER_PRM)
    assert main(["--prm", str(prm), "--smiles", "CCO", "--rules", str(rules)]) == 2
    out, err = capfd.readouterr()
    assert not out and "priority and target must be integers" in json.loads(err)["error"]


@pytest.mark.parametrize("max_order", [2.0, 3.0, 4.0, True, "2"])
def test_fitting_rejects_non_integer_max_order(max_order):
    reference = TypedReference(prepare_molecule("ClCl"), ("A", "B"), (-0.2, 0.2))
    with pytest.raises(AutoCGENFFError, match="max_order must be an integer"):
        fit_increments([reference], max_order=max_order)


def test_charge_model_rejects_string_type_keys():
    with pytest.raises(AutoCGENFFError, match="Invalid increment entry"):
        ChargeModel({"AB": (0.2,)}, max_order=2)
    mol = prepare_molecule("ClCl")
    model = ChargeModel({("A", "B"): (0.2,)}, max_order=2)
    result = assign_charges(mol, ("A", "B"), (Term(Kind.BOND, (0, 1)),), model)
    assert result.charges == (-0.2, 0.2)


@pytest.mark.parametrize(
    "term",
    [
        Term(Kind.BOND, (0, 2)),
        Term(Kind.BOND, (-1, 1)),
        Term(Kind.BOND, (False, 1)),
        Term(Kind.BOND, (0.0, 1)),
        Term(Kind.BOND, (0, 99)),
        Term(Kind.BOND, (0,)),
        Term(Kind.ANGLE, (0, 1, 3)),
        Term(Kind.ANGLE, (0, 1, 0)),
        Term(Kind.PROPER, (3, 0, 2, 1)),
    ],
)
def test_charge_assignment_rejects_invalid_covalent_paths(term):
    mol = prepare_molecule("CCC")
    types = ("A", "B", "C", *("H",) * (mol.GetNumAtoms() - 3))
    model = ChargeModel({("A", "B"): (0.1,), ("A", "C"): (0.2,), ("H", "B"): (0.3,)})
    with pytest.raises(AutoCGENFFError, match=f"Invalid {term.kind} charge path"):
        assign_charges(mol, types, (term,), model)
