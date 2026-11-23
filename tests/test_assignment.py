import json
from math import fsum
from pathlib import Path

import pytest
from rdkit import Chem

from autocgenff import (
    AssignmentOptions,
    AutoCGENFFError,
    PenaltyModel,
    parameterize,
    prepare_molecule,
    read_parameters,
    read_topology,
)
from autocgenff.assignment import _lookup_term
from autocgenff.cli import main
from autocgenff.molecule import Kind
from autocgenff.topology import improper_source_centers, match_template

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_forcefield(tmp_path: Path) -> tuple[Path, Path]:
    prm = tmp_path / "synthetic.prm"
    rtf = tmp_path / "synthetic.rtf"
    prm.write_text(
        """BONDS
O H 275 0.99
ANGLES
H O H 33 105
NONBONDED
O 0 -0.11 1.73
H 0 -0.03 0.31
END
""",
        encoding="utf-8",
    )
    rtf.write_text(
        """36 1
MASS 1 O 16 O
MASS 2 H 1 H
AUTO ANGLES DIHE
RESI SYN 0
ATOM H2 H 0.38
ATOM O1 O -0.76
ATOM H1 H 0.38
BOND O1 H1 O1 H2
RESI ASYM 0
ATOM O1 O -0.76
ATOM H1 H 0.36
ATOM H2 H 0.40
BOND O1 H1 O1 H2
RESI LP 0
ATOM O1 O -0.76
ATOM H1 H 0.38
ATOM H2 H 0.38
BOND O1 H1 O1 H2
LONEPAIR COLINEAR LP1 H1 O1 DIST 1.0
RESI WRONG 0
ATOM O1 O -0.76
ATOM H1 H 0.38
ATOM H2 H 0.38
BOND O1 H1
DOUBLE O1 H2
END
""",
        encoding="utf-8",
    )
    return prm, rtf


def test_named_template_remaps_and_emits_no_missing_terms(synthetic_forcefield):
    prm, rtf = synthetic_forcefield
    result = parameterize(
        "[H]O", read_parameters(prm), AssignmentOptions(topology=read_topology(rtf), template="syn")
    )
    assert result.is_complete and result.charge_source == "named_rtf_template"
    assert [atom.partial_charge for atom in result.atoms] == [0.38, -0.76, 0.38]
    assert result.atoms[1].template_name == "O1"
    assert any(term.term.kind == Kind.ANGLE for term in result.terms)


def test_named_template_checks_ambiguity_element_mapping_orders_and_virtual_sites(
    synthetic_forcefield,
):
    _, rtf = synthetic_forcefield
    topology = read_topology(rtf)
    molecule = prepare_molecule("O")
    with pytest.raises(AutoCGENFFError, match="Ambiguous"):
        match_template(molecule, topology, "ASYM")
    assigned = match_template(molecule, topology, "ASYM", atom_map={"O1": 0, "H1": 2, "H2": 1})
    assert [atom.charge for atom in assigned.atoms] == [-0.76, 0.40, 0.36]
    with pytest.raises(AutoCGENFFError, match="elements, connectivity and bond orders"):
        match_template(molecule, topology, "SYN", atom_map={"O1": 1, "H1": 2, "H2": 0})
    with pytest.raises(AutoCGENFFError, match="bond orders"):
        match_template(molecule, topology, "WRONG")
    with pytest.raises(AutoCGENFFError, match="LONEPAIR"):
        match_template(molecule, topology, "LP")


def test_cli_json_complete_incomplete_and_input_error(synthetic_forcefield, tmp_path, capsys):
    prm, rtf = synthetic_forcefield
    common = ["--prm", str(prm), "--smiles", "O"]
    assert main([*common, "--rtf", str(rtf), "--template", "SYN"]) == 0
    assert json.loads(capsys.readouterr().out)["is_complete"] is True
    rules = tmp_path / "rules.json"
    rules.write_text(
        '[{"name":"oxygen","smarts":"[#8]","atom_type":"O"},'
        '{"name":"hydrogen","smarts":"[#1]","atom_type":"H"}]',
        encoding="utf-8",
    )
    assert main([*common, "--rules", str(rules), "--no-impropers"]) == 2
    assert "toppar_water_ions.str" in json.loads(capsys.readouterr().err)["error"]
    assert main(["--prm", str(prm), "--smiles", "CO", "--rules", str(rules), "--no-impropers"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["is_complete"] is False
    assert all(atom["partial_charge"] is None for atom in output["atoms"])
    assert main([*common, "--rtf", str(rtf), "--template", "WRONG"]) == 2
    assert "bond orders" in json.loads(capsys.readouterr().err)["error"]


@pytest.fixture(scope="module")
def local_forcefield():
    prm = ROOT / "force_fields" / "par_all36_cgenff.prm"
    rtf = ROOT / "force_fields" / "top_all36_cgenff.rtf"
    if not prm.is_file() or not rtf.is_file():
        pytest.skip("User-supplied CGenFF v5 PRM/RTF not present")
    parameters = read_parameters(prm)
    topology = read_topology(rtf)
    assert len(topology.residues) == 2331
    assert len(tuple(parameters.iter_groups(Kind.NBFIX))) > 10
    return parameters, topology


@pytest.mark.parametrize(
    ("smiles", "name", "reference", "charge"),
    [
        ("CCO", "ETOH", None, 0),
        ("c1ccccc1", "BENZ", "c1ccccc1", 0),
        ("C[NH3+]", "MAMM", "C[NH3+]", 1),
        ("CC(=O)[O-]", "ACET", "CC(=O)[O-]", -1),
    ],
)
def test_real_v5_named_reference(local_forcefield, smiles, name, reference, charge):
    parameters, topology = local_forcefield
    result = parameterize(
        smiles,
        parameters,
        AssignmentOptions(topology=topology, template=name, template_smiles=reference),
    )
    assert result.is_complete
    assert fsum(atom.partial_charge for atom in result.atoms) == pytest.approx(charge)
    if name == "ACET":
        assert any(term.term.kind == Kind.IMPROPER and term.match for term in result.terms)


def test_real_v5_benzene_demands_identity_and_rejects_wrong_identity(local_forcefield):
    _, topology = local_forcefield
    benzene = prepare_molecule("c1ccccc1")
    with pytest.raises(AutoCGENFFError, match="requires template SMILES"):
        match_template(benzene, topology, "BENZ")
    with pytest.raises(AutoCGENFFError, match="differs"):
        match_template(benzene, topology, "BENZ", reference_smiles="C1CCCCC1")


def test_real_v5_sah_improper_has_a_center_last_parameter(local_forcefield):
    prm, topology = local_forcefield
    sah = topology.residues["SAH"]
    path = next(path for path in sah.impropers if path == ("OT1", "CA", "OT2", "C"))
    types_by_name = {atom.name: atom.atom_type for atom in sah.atoms}
    source = tuple(types_by_name[name] for name in path)
    assert prm.match(Kind.IMPROPER, source).records[0].types == source
    centers = improper_source_centers(topology)
    assert centers[source] == 3 and centers[source[::-1]] == 0
    target = ("Z", *source[1:])
    penalties = PenaltyModel(bonded={("Z", source[0]): 1})
    assert _lookup_term(prm, Kind.IMPROPER, target, penalties, 0, centers) == (
        "missing",
        None,
        None,
    )


def test_real_v5_2boh_does_not_verify_or_stereo_group(local_forcefield):
    _, topology = local_forcefield
    absolute = "CC[C@H](C)O"
    assert match_template(
        prepare_molecule(absolute), topology, "2BOH", reference_smiles=absolute
    ).verified_reference_smiles
    with pytest.raises(AutoCGENFFError, match="Enhanced stereochemistry"):
        match_template(
            prepare_molecule(Chem.MolFromSmiles(absolute + " |o1:2|")),
            topology,
            "2BOH",
            reference_smiles=absolute,
        )


def test_real_v5_cli_caller_increment_model(local_forcefield, capsys, tmp_path):
    parameters, _ = local_forcefield
    # Synthetic bond increments exercise the caller-supplied JSON interface.
    model = {
        "max_order": 2,
        "entries": [
            {"types": ["CG331", "CG321"], "values": [0.01]},
            {"types": ["CG321", "OG311"], "values": [0.29]},
            {"types": ["CG331", "HGA3"], "values": [0.08]},
            {"types": ["CG321", "HGA2"], "values": [0.08]},
            {"types": ["OG311", "HGP1"], "values": [0.39]},
        ],
    }
    increments = tmp_path / "increments.json"
    increments.write_text(json.dumps(model), encoding="utf-8")
    code = main(
        [
            "--prm",
            str(parameters.source),
            "--smiles",
            "CCO",
            "--no-impropers",
            "--increments",
            str(increments),
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["charge_source"] == "caller_increment_model"
    assert fsum(atom["partial_charge"] for atom in data["atoms"]) == pytest.approx(0)
