"""Light checks on the local benchmark tool (no LigandFF or force fields needed)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "benchmark_v5_local.py"


def _load():
    spec = __import__("importlib.util").util.spec_from_file_location("benchmark_v5_local", TOOL)
    module = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_records_definitions_and_checksums(tmp_path):
    module = _load()
    ff = tmp_path / "force_fields"
    ff.mkdir()
    for name in ("top_all36_cgenff.rtf", "par_all36_cgenff.prm", "par_all36_cgenff_increments.prm"):
        (ff / name).write_text(f"fake {name}\n")
    ligandff = tmp_path / "ligandff"
    (ligandff / "benchmarks" / "charmm_gui_cpa" / "raw").mkdir(parents=True)
    (ligandff / "benchmarks" / "b1_rtf_zinc.split.json").write_text("{}")
    (ligandff / "tests" / "fixtures").mkdir(parents=True)
    (ligandff / "tests" / "fixtures" / "si_v5_reference.json").write_text("[]")
    manifest = module._manifest(ligandff, ff)
    assert set(manifest["benchmarks"]) == {
        "B0", "B1", "B2", "B3", "B4", "QT", "QF", "QI", "OUT"
    }
    assert manifest["benchmarks"]["B1"]["splits"] == {"train": 904, "test": 226}
    sums = manifest["input_checksums"]["force_fields"]
    assert set(sums) == {
        "top_all36_cgenff.rtf", "par_all36_cgenff.prm", "par_all36_cgenff_increments.prm"
    }
    assert all(len(v) == 64 for v in sums.values())
    json.dumps(manifest)


def test_help_runs_without_ligandff():
    run = subprocess.run(
        [sys.executable, str(TOOL), "--help"], capture_output=True, text=True
    )
    assert run.returncode == 0
    assert "--ligandff" in run.stdout


@pytest.mark.parametrize(
    "kind,found,source,expected",
    [
        ("bond", ("A", "B"), ("A", "B"), True),
        ("bond", ("B", "A"), ("A", "B"), True),
        ("bond", ("A", "C"), ("A", "B"), False),
        ("improper", ("C", "A", "B", "D"), ("C", "B", "A", "D"), True),
        ("improper", ("A", "A", "B", "D"), ("C", "A", "B", "D"), False),
    ],
)
def test_same_source_b4_rule(kind, found, source, expected):
    module = _load()
    assert module._same_source(kind, found, source) is expected
