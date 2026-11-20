"""Synthetic regressions for the split adapter and train-derived typing rules."""

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from rdkit import Chem

from autocgenff.v5 import type_v5
from tools import score_v5_b1_split
from tools.b1_structure import StructureInput, prepare_structure


def _explicit(smiles: str) -> Chem.Mol:
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize(
    ("source", "target", "repaired", "method"),
    [
        ("[O]C=C(C)C(=O)OC", -1, "[O-]C=C(C)C(=O)OC", "radical-to-oxide"),
        ("[O-]C=C(C)[C]([O-])OC", -1, "[O-]C=C(C)C(=O)OC", "restore-carbonyl"),
        ("[O-][N-]CC", 0, "O=NCC", "restore-nitroso"),
        ("CS([O-])([O-])[O-]", -1, "CS(=O)(=O)[O-]", "restore-sulfonate"),
    ],
)
def test_local_repairs_preserve_indices_and_existing_pi_bonds(
    source: str, target: int, repaired: str, method: str,
) -> None:
    raw = _explicit(source)
    expected = _explicit(repaired)
    before = Chem.MolToSmiles(raw)
    result = prepare_structure(raw, raw, StructureInput(target))
    assert result.source == method
    fixed = result.molecule
    assert fixed is not None
    assert Chem.MolToSmiles(raw) == before
    assert Chem.GetFormalCharge(fixed) == target
    assert score_v5_b1_split._indexed_connectivity(fixed) == (
        score_v5_b1_split._indexed_connectivity(raw)
    )
    assert [a.GetFormalCharge() for a in fixed.GetAtoms()] == [
        a.GetFormalCharge() for a in expected.GetAtoms()
    ]
    assert [b.GetBondType() for b in fixed.GetBonds()] == [
        b.GetBondType() for b in expected.GetBonds()
    ]


def test_carbonyl_repair_allows_aromaticity_reperception() -> None:
    raw = _explicit("[C]1([O-])C=CC=C[NH]1")
    result = prepare_structure(raw, raw, StructureInput(0))

    assert result.source == "restore-carbonyl"
    fixed = result.molecule
    assert fixed is not None
    assert score_v5_b1_split._indexed_connectivity(fixed) == (
        score_v5_b1_split._indexed_connectivity(raw)
    )
    assert Chem.GetFormalCharge(fixed) == 0
    assert raw.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.SINGLE
    assert fixed.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.DOUBLE
    assert not any(b.GetIsAromatic() for b in raw.GetBonds())
    assert sum(b.GetIsAromatic() for b in fixed.GetBonds()) == 6


def test_keeps_charge_consistent_smiles_before_pinned_fallback() -> None:
    valid = _explicit("[O-]C=C(C)C(=O)OC")
    pinned = _explicit("O=CC(C)=C([O-])OC")
    result = prepare_structure(valid, pinned, StructureInput(-1))
    assert result.molecule is valid
    assert result.source == "unchanged-smiles"


@pytest.mark.parametrize(
    ("smiles", "charge"),
    [
        ("[O]C=C(C)C(=O)OC", 0),
        ("[C](C)([O-])[O-]", -1),
        ("[O-][N-]OC", 0),
        ("[O-][NH-]", 0),
        ("CS([18O-])([17O-])[16O-]", -1),
        ("CS([16O-])([17O-])[18O-]", -1),
    ],
)
def test_rejects_unsupported_charge_and_radical_repairs(smiles: str, charge: int) -> None:
    raw = _explicit(smiles)
    result = prepare_structure(raw, raw, StructureInput(charge))
    assert result.molecule is None
    assert result.source == "unsupported-repair"


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("O=NCC", {0: "OG2D1", 1: "NG2D1"}),
        ("C[N+](=O)[O-]", {1: "NG2O1", 2: "OG2N1", 3: "OG2N1"}),
        ("C[N+](C)(C)[O-]", {4: "OG312"}),
        ("O=C1NS(=O)(=O)C=C1", {1: "CG2R53", 2: "NG2R53"}),
        ("O=C1N(C)C=CC1=O", {2: "NG2R51"}),
        ("CC=C1CCCC1", {1: "CG2D1", 2: "CG2D1"}),
        ("C=CC=C1CCCC1", {2: "CG2DC2", 3: "CG25C2"}),
        ("C=C1CCCN1", {1: "CG251O"}),
    ],
)
def test_train_motif_on_independent_smiles(smiles: str, expected: dict[int, str]) -> None:
    result = type_v5(smiles)
    assert result.is_complete
    assert {i: result.types[i] for i in expected} == expected


@pytest.mark.parametrize(
    ("smiles", "index", "expected"),
    [
        ("CC1(C)CCCC(C)(C)[N+]1=O", 10, "OG2N1"),
        ("C#CC=C1CCCC1", 3, "CG25C1"),
        ("COC=C1CCCC1", 3, "CG25C1"),
        ("c1ccccc1C=C2CCCC2", 7, "CG25C1"),
        ("O=C1C=CC2N1S(=O)(=O)CC=2", 5, "NG2R51"),
        # Both rings individually pass the ring2 test but only one contains the carbonyl bond.
        ("O=C1C=CC2N1S(=O)(=O)C(=O)C=2", 5, "NG2R51"),
        ("O=C1[N-]S(=O)(=O)C=C1", 2, "NG2R51"),
        ("S=C1NS(=O)(=O)C=C1", 2, "NG2R51"),
    ],
)
def test_train_rules_preserve_neighboring_legacy_motifs(
    smiles: str, index: int, expected: str,
) -> None:
    assert type_v5(smiles).types[index] == expected


def test_preflight_refuses_non_utf8_default_before_opening_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import TextIOWrapper

    def cp1252_text_io(buffer):
        return TextIOWrapper(buffer, encoding="cp1252")

    monkeypatch.setattr(score_v5_b1_split, "TextIOWrapper", cp1252_text_io)
    with pytest.raises(ValueError, match="requires UTF-8 text I/O"):
        score_v5_b1_split.score_split(tmp_path / "absent.rtf", tmp_path / "absent.json", "train", (1, 1, 0))


def test_preflight_refuses_mixed_manifest_before_opening_rtf(tmp_path: Path) -> None:
    manifest = tmp_path / "mixed.json"
    manifest.write_text(json.dumps({"train": ["A"], "other": ["B"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest must contain only"):
        score_v5_b1_split.score_split(tmp_path / "absent.rtf", manifest, "train", (1, 1, 0))


@pytest.mark.parametrize("header", ["RESI", "RESI A"])
def test_preflight_rejects_truncated_residue_header(header: str) -> None:
    with pytest.raises(ValueError, match="RTF line 1: RESI requires a name and net charge"):
        score_v5_b1_split._structure_inputs(f"{header}\nEND\n".encode())


def test_scoring_reports_invalid_aligned_smiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rtf = tmp_path / "train.rtf"
    rtf.write_text("RESI A 0\nEND\n", encoding="utf-8")
    manifest = tmp_path / "train.json"
    manifest.write_text('{"train": ["A"]}', encoding="utf-8")
    monkeypatch.setattr(score_v5_b1_split, "TRAIN_RTF_SHA256", hashlib.sha256(rtf.read_bytes()).hexdigest())

    molecule = _explicit("C")
    case = SimpleNamespace(name="A", mol=molecule, types=type_v5(molecule).types, smiles="invalid_")
    alignment = {"molecules": 1, "atoms": 5, "repaired_radicals": 0, "skipped": {}}
    benchmarks = ModuleType("ligandff.benchmarks")
    benchmarks.b1 = SimpleNamespace(build=lambda *_args, **_kwargs: ([case], alignment))
    monkeypatch.setitem(sys.modules, "ligandff.benchmarks", benchmarks)

    with pytest.raises(ValueError, match="B1 SMILES could not be parsed for A"):
        score_v5_b1_split.score_split(rtf, manifest, "train", (1, 5, 0))


def test_preflight_refuses_extra_residues_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({"train": ["A"]}), encoding="utf-8")
    rtf = tmp_path / "synthetic.rtf"
    rtf.write_text("RESI A 0.0\nRESI B 0.0\nEND\n", encoding="utf-8")
    checksum = hashlib.sha256(rtf.read_bytes()).hexdigest()
    monkeypatch.setattr(score_v5_b1_split, "TRAIN_RTF_SHA256", checksum)
    with pytest.raises(ValueError, match="refusing to build"):
        score_v5_b1_split.score_split(rtf, manifest, "train", (1, 1, 0))


@pytest.mark.parametrize("card", ["DOUBLE", "TRIPLE", "DOUB", "TRIP"])
def test_preflight_refuses_unverifiable_explicit_order_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, card: str,
) -> None:
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({"train": ["DUMMY"]}), encoding="utf-8")
    rtf = tmp_path / "synthetic.rtf"
    rtf.write_text(f"RESI DUMMY 0.0\n{card} C1 C2\nEND\n", encoding="utf-8")
    checksum = hashlib.sha256(rtf.read_bytes()).hexdigest()
    monkeypatch.setattr(score_v5_b1_split, "TRAIN_RTF_SHA256", checksum)
    with pytest.raises(ValueError, match="DUMMY: explicit RTF bond-order cards are not supported"):
        score_v5_b1_split.score_split(rtf, manifest, "train", (1, 2, 0))


def test_evaluation_preflight_requires_independent_verification(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="independently verified evaluation inputs"):
        score_v5_b1_split.score_split(
            tmp_path / "absent.rtf", tmp_path / "absent.json", "test", (1, 1, 0),
        )


@pytest.mark.parametrize(
    ("name", "pin_train_manifest", "rtf_payload", "error"),
    [
        ("TRAIN", True, "RESI TRAIN 0.0\nEND\n", "overlaps verified train membership"),
        ("DUMMY", False, None, "expected the pinned train manifest"),
        ("DUMMY", True, "RESI DUMMY 0.0\nEND\n", "RTF SHA-256 does not match"),
        ("DUMMY", True, "RESI TRAIN 0.0\nEND\n", "RTF residues do not match"),
    ],
)
def test_evaluation_preflight_rejects_unverified_synthetic_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
    pin_train_manifest: bool, rtf_payload: str | None, error: str,
) -> None:
    train = tmp_path / "train.json"
    train.write_text(json.dumps({"train": ["TRAIN"]}), encoding="utf-8")
    if pin_train_manifest:
        monkeypatch.setattr(score_v5_b1_split, "TRAIN_COUNTS", (1, 1, 0))
        monkeypatch.setattr(
            score_v5_b1_split, "TRAIN_MANIFEST_SHA256", hashlib.sha256(train.read_bytes()).hexdigest(),
        )
    manifest = tmp_path / "synthetic.json"
    manifest.write_text(json.dumps({"test": [name]}), encoding="utf-8")
    rtf = tmp_path / "synthetic.rtf"
    if rtf_payload is not None:
        rtf.write_text(rtf_payload, encoding="utf-8")
    # Use the checksum of the synthetic train RTF, including for the relabeling attempt.
    train_checksum = hashlib.sha256(b"RESI TRAIN 0.0\nEND\n").hexdigest()
    provenance = score_v5_b1_split.FrozenEvaluationInput(train_checksum, train)
    with pytest.raises(ValueError, match=error):
        score_v5_b1_split.score_split(rtf, manifest, "test", (1, 1, 0), evaluation=provenance)


def test_evaluation_preflight_accepts_matching_disjoint_synthetic_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = tmp_path / "train.json"
    train.write_text('{"train": ["TRAIN"]}', encoding="utf-8")
    monkeypatch.setattr(score_v5_b1_split, "TRAIN_COUNTS", (1, 1, 0))
    monkeypatch.setattr(
        score_v5_b1_split, "TRAIN_MANIFEST_SHA256", hashlib.sha256(train.read_bytes()).hexdigest(),
    )
    manifest = tmp_path / "synthetic.json"
    manifest.write_text('{"test": ["DUMMY"]}', encoding="utf-8")
    rtf = tmp_path / "synthetic.rtf"
    rtf.write_text("RESI DUMMY 0.0\nEND\n", encoding="utf-8")
    digest = hashlib.sha256(rtf.read_bytes()).hexdigest()
    provenance = score_v5_b1_split.FrozenEvaluationInput(digest, train)

    verified = score_v5_b1_split._verify_inputs(rtf, manifest, "test", (1, 1, 0), provenance)

    assert verified.metadata == {"DUMMY": StructureInput(0)}


@pytest.mark.parametrize(
    ("mismatch", "replace"),
    [(False, None), (True, None), (True, "rtf"), (True, "manifest")],
    ids=["exact", "one-wrong-indexed-type", "rtf-replaced", "manifest-replaced"],
)
def test_exact_cli_gate_on_synthetic_b1_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    mismatch: bool, replace: str | None,
) -> None:
    b1 = pytest.importorskip("ligandff.benchmarks.b1")
    smiles = "C=CC=CC"
    mol = _explicit(smiles)
    # Explicitly specified expectation; do not generate the labels using the scorer under test.
    types = ["CG2DC3", "CG2DC1", "CG2DC2", "CG2DC2", "CG331", "HGA5", "HGA5",
             "HGA4", "HGA4", "HGA4", "HGA3", "HGA3", "HGA3"]
    if mismatch:
        types[1] = "CG2DC2"
    rows = ["RESI DIENE 0", f"! SMILES: {smiles}"]
    rows.extend(f"ATOM A{i} {label} 0.0" for i, label in enumerate(types))
    rows.extend(f"BOND A{b.GetBeginAtomIdx()} A{b.GetEndAtomIdx()}" for b in mol.GetBonds())
    rtf = tmp_path / "synthetic.rtf"
    rtf.write_text("\n".join([*rows, "END", ""]), encoding="utf-8")
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({"train": ["DIENE"]}), encoding="utf-8")
    checksum = hashlib.sha256(rtf.read_bytes()).hexdigest()
    monkeypatch.setattr(score_v5_b1_split, "TRAIN_RTF_SHA256", checksum)
    monkeypatch.setattr(score_v5_b1_split, "TRAIN_COUNTS", (1, 13, 0))
    if replace is not None:
        pinned_build = b1.build

        def build_after_replacing_input(rtf_path: Path, *, manifest: Path) -> tuple:
            if replace == "rtf":
                corrected = rtf.read_text(encoding="utf-8").replace("A1 CG2DC2", "A1 CG2DC1")
                rtf.write_text(corrected, encoding="utf-8")
            else:
                (tmp_path / "train.json").write_text('{"train": ["OTHER"]}', encoding="utf-8")
            return pinned_build(rtf_path, manifest=manifest)

        monkeypatch.setattr(b1, "build", build_after_replacing_input)
    monkeypatch.setattr(sys, "argv", [
        "score_v5_b1_split", str(rtf), "--manifest", str(manifest), "--split", "train",
        "--details", "--require-exact",
    ])

    assert score_v5_b1_split.main() == int(mismatch)
    output = json.loads(capsys.readouterr().out)["scores"]["structure-corrected"]
    assert output["counts"] == {
        "atoms": 13, "molecules": 1, "exact_atoms": 13 - int(mismatch),
        "exact_molecules": int(not mismatch), "rejected_molecules": 0, "incomplete_molecules": 0,
    }
    assert output["structure_sources"] == {"unchanged-smiles": 1}
    if mismatch:
        assert output["misses"]["DIENE"] == [
            {"index": 1, "predicted": "CG2DC1", "reference": "CG2DC2",
             "neighbors": [[0, "C"], [2, "C"], [7, "H"]]},
        ]
    else:
        assert output["misses"] == {}
