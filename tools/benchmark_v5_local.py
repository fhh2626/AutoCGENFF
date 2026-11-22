#!/usr/bin/env python3
"""Reproducible local benchmark suite for AutoCGENFF local-v5.

Builds the LigandFF reference sets (B0/B1/B2/B3/B4 plus derived QT/QF/QI/OUT)
from the pinned LigandFF checkout and the caller's local force-field files,
then scores THIS repository's code. Aggregate results only in the output; no
per-case details are kept so test splits stay sealed.

Run inside the repo, e.g.:

    PYTHONPATH=/path/to/pinned-ligandff/src:src \
        python tools/benchmark_v5_local.py \
        --ligandff /path/to/pinned-ligandff \
        --output benchmark_local/baseline.json

--write-manifest records input checksums, split manifests and benchmark
definitions under benchmark_local/ (gitignored; raw force-field data is never
copied). Requires the local RTF/PRM/increment files, which stay private.
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
LIGANDFF_COMMIT = "860b17da7e982ef7ab62d993a0e24fda3958bbb6"
SETS = frozenset("B0 B1 B2 B3 B4 QT QF QI OUT".split())

sys.path.insert(0, str(REPO / "src"))

from rdkit import Chem, rdBase  # noqa: E402

from autocgenff import V5ForceField, parameterize_v5  # noqa: E402
from autocgenff.analogy import CandidateIndex, best_analogy  # noqa: E402
from autocgenff.charges import ZERO_WEIGHT, assign_charges, reverse_increments  # noqa: E402
from autocgenff.errors import AutoCGENFFError  # noqa: E402
from autocgenff.molecule import Kind, inventory  # noqa: E402
from autocgenff.parameters import read_parameters  # noqa: E402
from autocgenff.topology import read_topology  # noqa: E402
from autocgenff.v5 import _prepare_v5, _type_prepared, type_v5  # noqa: E402
from autocgenff.v5_inputs import observed_v5_penalty_model  # noqa: E402
from autocgenff.v5_stream import write_v5_stream  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ligandff_head(path: Path) -> str:
    marker = path / ".source-commit"
    if marker.is_file():
        return marker.read_text(encoding="ascii").strip()
    if not (path / ".git").exists():
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _manifest(ligandff: Path, ff_dir: Path) -> dict:
    inputs = {
        "force_fields": {
            name: _sha256(ff_dir / name)
            for name in (
                "top_all36_cgenff.rtf",
                "par_all36_cgenff.prm",
                "par_all36_cgenff_increments.prm",
            )
            if (ff_dir / name).is_file()
        },
        "ligandff_split_manifests": {
            name: _sha256(ligandff / "benchmarks" / name)
            for name in (
                "b1_rtf_zinc.split.json",
                "b2_rtf_legacy.split.json",
                "b4_annotations.split.json",
            )
            if (ligandff / "benchmarks" / name).is_file()
        },
        "ligandff_builder_code": {
            str(item.relative_to(ligandff)): _sha256(item)
            for item in sorted((ligandff / "src" / "ligandff").rglob("*.py"))
        },
        "ligandff_fixtures": {
            "tests/fixtures/si_v5_reference.json": _sha256(
                ligandff / "tests" / "fixtures" / "si_v5_reference.json"
            )
        },
        "b3_raw_topologies": {
            item.name: _sha256(item)
            for item in sorted(
                (ligandff / "benchmarks" / "charmm_gui_cpa" / "raw").glob("*.itp")
            )
        },
    }
    return {
        "schema": "autocgenff.benchmark_local.manifest.v1",
        "repo": str(REPO),
        "ligandff_path": str(ligandff),
        "ligandff_commit": LIGANDFF_COMMIT,
        "rdkit_version": rdBase.rdkitVersion,
        "force_field_dir": str(ff_dir),
        "benchmarks": {
            "B0": {"definition": "CGenFF 5.0 program SI Table S10; charge/penalty at 0.001", "splits": {"reference": 3}},
            "B1": {"definition": "RTF residues with SMILES; types+charges", "splits": {"train": 904, "test": 226}},
            "B2": {"definition": "RTF residues without SMILES; types+charges", "splits": {"train": 728, "test": 182}},
            "B3": {"definition": "CHARMM-GUI topologies (CGenFF<=4.6)", "splits": {"dev": 34}},
            "B4": {"definition": "PRM analogies; train-only costs; B4 pool", "splits": {"train": 5787, "test": 1447}},
            "QT": {"definition": "increment-only fit vs RTF charges, zero seeds", "splits": {"all": "residues"}},
            "QF": {"definition": "rule_charge vs inferred integer seed", "splits": {"derived": "B1/B2"}},
            "QI": {"definition": "impr centers vs RTF IMPR first atoms", "splits": {"derived": "B1/B2"}},
            "OUT": {"definition": "parameterize+stream success; OpenMM energy N/A locally", "splits": {"derived": "B1/B2/B3"}},
        },
        "input_checksums": inputs,
        "notes": [
            "B1-test molecule 23001B was excluded upstream in earlier ad-hoc runs; this suite reports the full 226-case build.",
            "B4 excludes the target and, for batch cases, all parameters from the batch.",
            "Aggregates only; no per-case results are written.",
        ],
    }


def _empty() -> dict:
    return {
        "molecules": 0,
        "molecules_exact": 0,
        "atoms": 0,
        "atoms_exact": 0,
        "errors": 0,
    }


def _canonical(t: str | None) -> str | None:
    from ligandff.benchmarks.common import canonical_type

    return None if t is None else canonical_type(t)


def score_typing(cases) -> dict:
    """Strict index-aligned types and canonical (altnum-folded) types per split."""
    splits = {}
    for case in cases:
        row = splits.setdefault(
            case.split,
            {"typing": _empty(), "typing_canonical": _empty()},
        )
        for counts in row.values():
            counts["molecules"] += 1
            counts["atoms"] += len(case.types)
        try:
            with rdBase.BlockLogs():
                got = type_v5(case.mol).types
        except AutoCGENFFError:
            got = ()
        if len(got) != len(case.types):
            for counts in row.values():
                counts["errors"] += 1
            continue
        strict = sum(a == b for a, b in zip(case.types, got, strict=True))
        canonical = sum(_canonical(a) == _canonical(b) for a, b in zip(case.types, got, strict=True))
        for counts, matches in zip(row.values(), (strict, canonical), strict=True):
            counts["atoms_exact"] += matches
            counts["molecules_exact"] += matches == len(case.types)
    return {k: v["typing"] | {"canonical": v["typing_canonical"]} for k, v in sorted(splits.items())}


def score_charges(cases, ff: V5ForceField) -> dict:
    """Partial charges vs reference: <0.1 e pass, unresolved = fail."""
    splits = {}
    for case in cases:
        ref = case.meta.get("charges")
        if ref is None or case.meta.get("lonepair"):
            continue
        row = splits.setdefault(
            case.split,
            {
                "molecules": 0,
                "molecules_exact": 0,
                "molecules_resolved": 0,
                "atoms": 0,
                "atoms_within_0.1": 0,
                "unresolved": 0,
                "errors": 0,
                "_rmse": [],
            },
        )
        row["molecules"] += 1
        row["atoms"] += len(ref)
        try:
            with rdBase.BlockLogs():
                molecule = _prepare_v5(case.mol)
                typed = _type_prepared(molecule)
                if not typed.is_complete:
                    row["errors"] += 1
                    continue
                result = assign_charges(
                    molecule, typed.types, inventory(molecule), ff.charges,
                    penalties=ff.penalties, baseline=tuple(a.rule_charge for a in typed.atoms),
                )
        except AutoCGENFFError:
            row["errors"] += 1
            continue
        charges = result.charges
        if charges is None or len(charges) != len(ref):
            row["unresolved"] += 1
            continue
        row["molecules_resolved"] += 1
        deltas = [abs(p - q) for p, q in zip(charges, ref)]
        row["atoms_within_0.1"] += sum(d < 0.1 for d in deltas)
        row["molecules_exact"] += all(d < 0.1 for d in deltas)
        row["_rmse"].append(math.sqrt(sum(d * d for d in deltas) / len(deltas)))
    for row in splits.values():
        rmses = row.pop("_rmse")
        row["median_rmse"] = statistics.median(rmses) if rmses else None
        row["atoms_within_0.1_fraction"] = (
            row["atoms_within_0.1"] / row["atoms"] if row["atoms"] else None
        )
    return splits


def score_qf(cases, inc_table) -> dict:
    """rule_charge seeds vs inferred integer q0 (QF, official eligibility)."""
    from ligandff.benchmarks.qf import inferred_formal_charges

    splits = {}
    for case in cases:
        if case.meta.get("lonepair") or case.meta.get("charges") is None:
            continue
        q0 = inferred_formal_charges(case, inc_table)
        if q0 is None:
            continue
        try:
            with rdBase.BlockLogs():
                typed = type_v5(Chem.Mol(case.mol))
        except AutoCGENFFError:
            continue
        if len(typed.atoms) != len(q0) or tuple(map(_canonical, typed.types)) != tuple(
            map(_canonical, case.types)
        ):
            continue
        row = splits.setdefault(case.split, _empty())
        correct = sum(a.rule_charge == b for a, b in zip(typed.atoms, q0, strict=True))
        row["molecules"] += 1
        row["molecules_exact"] += correct == len(q0)
        row["atoms"] += len(q0)
        row["atoms_exact"] += correct
    return dict(sorted(splits.items()))


def score_qi(cases) -> dict:
    """impr centers vs RTF IMPR first atoms; eligible = canonically exact typing."""
    splits = {}
    for case in cases:
        try:
            with rdBase.BlockLogs():
                typed = type_v5(case.mol)
        except AutoCGENFFError:
            continue
        if tuple(map(_canonical, typed.types)) != tuple(map(_canonical, case.types)):
            continue
        ref = {imp[0] for imp in case.meta.get("impropers", [])}
        got = {a.index for a in typed.atoms if a.improper_center}
        row = splits.setdefault(
            case.split,
            {"molecules": 0, "centers_exact": 0, "atoms": 0, "atoms_correct": 0},
        )
        row["molecules"] += 1
        row["centers_exact"] += ref == got
        row["atoms"] += len(case.types)
        row["atoms_correct"] += sum((i in ref) == (i in got) for i in range(len(case.types)))
    return dict(sorted(splits.items()))


def score_qt(rtf_path: Path, ff: V5ForceField) -> dict:
    """Increment-only charge fit on RTF residues with OUR native model (QT)."""
    from ligandff.benchmarks.common import element_of_type
    from ligandff.io.rtf import read_rtf

    eligible = ineligible = skipped = 0
    rmses, within_01, within_005, within_01e = [], 0, 0, 0
    atoms_total = residues_within = 0
    for res in read_rtf(rtf_path):
        if (
            res.kind != "RESI"
            or res.has_lonepair
            or res.has_external_bond
            or len(res.atoms) < 2
        ):
            continue
        elements = [element_of_type(a.type) for a in res.atoms]
        if any(e is None for e in elements):
            skipped += 1
            continue
        idx = {a.name: i for i, a in enumerate(res.atoms)}
        mol = Chem.RWMol()
        for element in elements:
            at = Chem.Atom(element)
            at.SetNoImplicit(True)
            mol.AddAtom(at)
        bonds = {
            tuple(sorted((idx[a], idx[b])))
            for a, b in res.bonds
            if a in idx and b in idx and a != b
        }
        for i, j in sorted(bonds):
            mol.AddBond(i, j, Chem.BondType.SINGLE)
        molecule = mol.GetMol()
        types = tuple(a.type for a in res.atoms)
        try:
            result = assign_charges(
                molecule, types, inventory(molecule), ff.charges,
                baseline=[0] * len(types),
            )
        except AutoCGENFFError:
            ineligible += 1
            continue
        n_a = sum(
            t.status in {"not_applicable", "linear_not_applicable"}
            for t in result.terms
        )
        if result.charges is None or n_a:
            ineligible += 1
            continue
        eligible += 1
        atoms_total += len(types)
        errors = [a.charge - q - round(a.charge - q) for a, q in zip(res.atoms, result.charges)]
        within_01 += sum(abs(e) < 0.01 for e in errors)
        within_005 += sum(abs(e) < 0.05 for e in errors)
        within_01e += sum(abs(e) < 0.1 for e in errors)
        residues_within += all(abs(e) < 0.1 for e in errors)
        rmses.append(math.sqrt(sum(e * e for e in errors) / len(errors)))
    return {
        "residues": eligible,
        "ineligible": ineligible,
        "untyped_elements": skipped,
        "median_rmse": statistics.median(rmses) if rmses else None,
        "atoms": atoms_total,
        "atoms_within_0.1": within_01e,
        "residues_within_0.1": residues_within,
        "fraction_within_0.01": within_01 / atoms_total if atoms_total else None,
        "fraction_within_0.05": within_005 / atoms_total if atoms_total else None,
        "fraction_within_0.1": within_01e / atoms_total if atoms_total else None,
    }


def _b0_molecule(fixture) -> Chem.Mol:
    from ligandff.benchmarks.common import element_of_type

    idx = {a["name"]: i for i, a in enumerate(fixture["atoms"])}
    mol = Chem.RWMol()
    for atom in fixture["atoms"]:
        element = element_of_type(atom["type"])
        if element is None:
            raise AutoCGENFFError(f"B0: unmapped element for {atom['type']}")
        at = Chem.Atom(element)
        at.SetNoImplicit(True)
        at.SetFormalCharge(int(round(fixture["formal_charges"].get(atom["name"], 0))))
        mol.AddAtom(at)
    for a, b in fixture["bonds"]:
        mol.AddBond(idx[a], idx[b], Chem.BondType.SINGLE)
    return mol.GetMol()


def score_b0(fixtures, ff: V5ForceField) -> dict:
    """Exact program output: charge and charge_penalty agreement at 0.001.

    Missing increments are resolved through the reference's printed analog
    (path-level source substitution with the printed penalty).
    """
    rows = []
    for fixture in fixtures:
        molecule = _b0_molecule(fixture)
        types = tuple(a["type"] for a in fixture["atoms"])
        analogs = {
            tuple(x["missing"]): (tuple(x["source"]), float(x["penalty"]))
            for x in fixture["analog_parameters"]
        }
        baseline = [
            float(fixture["formal_charges"].get(a["name"], 0))
            for a in fixture["atoms"]
        ]
        deltas = [[] for _ in types]
        errors = [[] for _ in types]
        missing = []
        for term in inventory(molecule):
            if term.kind not in {Kind.BOND, Kind.ANGLE, Kind.PROPER}:
                continue
            key = tuple(types[i] for i in term.atoms)
            canonical = min(key, key[::-1])
            score = 0.0
            oriented = key
            if canonical not in ff.charges.increments:
                hit = analogs.get(key) or analogs.get(key[::-1])
                if hit is None:
                    missing.append(key)
                    continue
                oriented = hit[0] if key in analogs else hit[0][::-1]
                canonical = min(oriented, oriented[::-1])
                score = hit[1]
            if canonical not in ff.charges.increments:
                missing.append(key)
                continue
            values = ff.charges.increments[canonical]
            if term.kind == Kind.PROPER and score > ff.charges.torsion_cutoff:
                values = (0.0, 0.0, 0.0)
                score = ff.charges.torsion_cutoff
            if values is None:
                continue
            if oriented != canonical:
                values = reverse_increments(values)
            extended = (0.0, *values, 0.0)
            for k, atom in enumerate(term.atoms):
                net = extended[k] - extended[k + 1]
                deltas[atom].append(net)
                errors[atom].append((abs(net) + ZERO_WEIGHT) ** (1 / 3) * score**2)
        if missing:
            rows.append({"name": fixture["name"], "atoms": len(types), "atoms_exact": 0,
                         "exact": False, "missing": [list(m) for m in missing]})
            continue
        charges = [math.fsum((b, *d)) for b, d in zip(baseline, deltas)]
        penalties = [math.sqrt(math.fsum(e)) for e in errors]
        dq = [abs(round(c, 3) - a["charge"]) for c, a in zip(charges, fixture["atoms"])]
        dp = [
            abs(round(p, 3) - a["charge_penalty"])
            for p, a in zip(penalties, fixture["atoms"])
        ]
        matches = sum(q < 1e-9 and p < 1e-9 for q, p in zip(dq, dp, strict=True))
        exact = matches == len(types) and abs(
            round(max(penalties), 3) - fixture["charge_penalty"]
        ) < 1e-9
        rows.append(
            {
                "name": fixture["name"],
                "atoms": len(types),
                "atoms_exact": matches,
                "exact": exact,
                "max_abs_dq_0.001": max(dq, default=0.0),
                "max_abs_dpenalty_0.001": max(dp, default=0.0),
            }
        )
    return {"molecules": len(rows), "molecules_exact": sum(r["exact"] for r in rows),
            "atoms": sum(r["atoms"] for r in rows), "atoms_exact": sum(r["atoms_exact"] for r in rows),
            "per_molecule": rows}


def _same_source(kind: str, found: tuple, source: tuple) -> bool:
    """B4 agreement: same source, allowing reversal and improper outer rotation."""
    if found in (source, source[::-1]):
        return True
    return kind == "improper" and found[0] == source[0] and sorted(found[1:]) == sorted(source[1:])


def score_b4(prm_path: Path, manifest: Path, ff: V5ForceField, work: Path) -> dict:
    """Score the Part II directional analogy profile fitted to B4-train annotations only."""
    from ligandff.benchmarks import b4
    from ligandff.params.database import ParameterDatabase

    kinds = {"bond": Kind.BOND, "angle": Kind.ANGLE,
             "dihedral": Kind.PROPER, "improper": Kind.IMPROPER}
    db = ParameterDatabase.from_prm(prm_path)
    cases = b4.build(db, manifest)
    train = {c.name for c in cases if c.split == "train"}
    allowed_lines = set()
    for case in cases:
        if case.name in train:
            match = ff.parameters.match(kinds[case.kind], case.types)
            if match is not None and not match.wildcard:
                allowed_lines.add(match.records[0].line)
    # Keep all mechanical records but hide all non-training annotation labels
    # from the cost estimator. The reported train result is in-sample.
    annotation = re.compile(r"\bfrom\s+[A-Z0-9 ]+?\s*,\s*penalty\s*=", re.I)
    with TemporaryDirectory(dir=work) as tmp:
        training_prm = Path(tmp) / prm_path.name
        with prm_path.open(encoding="utf-8-sig") as src, training_prm.open(
            "w", encoding="utf-8"
        ) as dst:
            for number, line in enumerate(src, 1):
                if number not in allowed_lines and annotation.search(line.partition("!")[2]):
                    line = line.partition("!")[0].rstrip() + "\n"
                dst.write(line)
        model, evidence = observed_v5_penalty_model(read_parameters(training_prm), ff.topology)
    if any(e.line not in allowed_lines for e in evidence):
        raise ValueError("B4 cost evidence escaped the training annotation filter")
    # Build exact inner-atom candidate indices for all and non-batch pools.
    all_idx = {}
    nonbatch_idx = {}
    for name, kind in kinds.items():
        batch_keys = {
            min(p.types, p.types[::-1]) for p in db.params[name].values() if b4.batch_era(p)
        }
        all_keys = [group[0].types for group in ff.parameters.iter_groups(kind)]
        nb_keys = [key for key in all_keys if min(key, key[::-1]) not in batch_keys]
        if kind == Kind.IMPROPER:
            for center in (0, 3):
                all_idx[(kind, center)] = CandidateIndex(
                    kind, all_keys, improper_center=center,
                    source_improper_centers=ff.improper_centers,
                )
                nonbatch_idx[(kind, center)] = CandidateIndex(
                    kind, nb_keys, improper_center=center,
                    source_improper_centers=ff.improper_centers,
                )
        else:
            all_idx[(kind, None)] = CandidateIndex(kind, all_keys)
            nonbatch_idx[(kind, None)] = CandidateIndex(kind, nb_keys)
    splits = {}
    for case in cases:
        kind = kinds[case.kind]
        center = ff.improper_centers.get(case.types) if kind == Kind.IMPROPER else None
        row = splits.setdefault(
            case.split,
            {"cases": 0, "analogy_found": 0, "same_source": 0, "tps_within_0.5": 0, "both": 0},
        )
        row["cases"] += 1
        if kind == Kind.IMPROPER and center not in (0, 3):
            continue
        requested = min(case.types, case.types[::-1])
        index = (nonbatch_idx if case.batch else all_idx)[(kind, center)]
        candidate = best_analogy(
            kind, case.types, (), model, improper_center=center,
            source_improper_centers=ff.improper_centers,
            candidate_index=index, exclude={requested},
        )
        if candidate is None:
            continue
        row["analogy_found"] += 1
        same = _same_source(case.kind, candidate.types, case.source)
        close = abs(candidate.penalty - case.penalty) <= 0.5 + 1e-9
        row["same_source"] += same
        row["tps_within_0.5"] += close
        row["both"] += same and close
    return {"pool": "B4 leave-one-out with batch exclusion", "cost_evidence": len(evidence),
            "splits": dict(sorted(splits.items()))}


def _readback(case, result, stream: Path, psfgen: Path, base_params) -> bool:
    """Build a PSF from the exported stream and test a finite OpenMM energy."""
    import openmm as mm
    from openmm import app, unit
    from rdkit.Chem import AllChem

    root = stream.parent
    psf = root / "ligand.psf"
    psf.unlink(missing_ok=True)
    base = Path(os.path.relpath(result.topology_source, root)).as_posix()
    script = (
        f"topology {{{base}}}\ntopology {stream.name}\n"
        "segment L { residue 1 LIG }\nwritepsf x-plor ligand.psf\nexit\n"
    )
    tcl = root / "generate.tcl"
    tcl.write_text(script, encoding="utf-8")
    run = subprocess.run([str(psfgen), tcl.name], cwd=root, capture_output=True, timeout=30)
    if run.returncode or not psf.is_file():
        return False
    parser = app.CharmmPsfFile(str(psf))
    parsed = list(parser.atom_list)
    expected_sites = [
        *((a.name, a.atom_type, a.partial_charge) for a in result.atoms),
        *((lp.name, lp.atom_type, lp.charge) for lp in result.lone_pairs),
    ]
    if len(parsed) != len(expected_sites) or any(
        str(a.attype) != atom_type or a.name != name
        or not math.isclose(a.charge, charge, abs_tol=5.1e-7, rel_tol=0.0)
        for a, (name, atom_type, charge) in zip(parsed, expected_sites, strict=True)
    ):
        return False
    params = base_params
    if "read param card" in stream.read_text(encoding="ascii"):
        params = copy(base_params)
        # Our stream can add bonded terms only. Keep the base tables pristine.
        for name in ("bond_types", "angle_types", "urey_bradley_types", "dihedral_types",
                     "improper_types"):
            setattr(params, name, getattr(base_params, name).copy())
        params.readStreamFile(str(stream))
    system = parser.createSystem(params, nonbondedMethod=app.NoCutoff, constraints=None)
    if system.getNumParticles() != len(expected_sites):
        return False
    n_real = len(result.atoms)
    if any(not system.isVirtualSite(idx) for idx in range(n_real, len(expected_sites))):
        return False
    mol = Chem.Mol(case.mol)
    with rdBase.BlockLogs():
        if AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True, maxAttempts=20) != 0:
            return False
    coords = [list(p) for p in mol.GetConformer().GetPositions()]
    coords.extend([[0.0, 0.0, 0.0]] * len(result.lone_pairs))
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(coords * unit.angstrom)
    if result.lone_pairs:
        context.computeVirtualSites()
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    return math.isfinite(energy.value_in_unit(unit.kilojoule_per_mole))


def score_out(cases, stream: Path, ff: V5ForceField, psfgen: Path | None = None) -> dict:
    """Measure stream export and optionally independent readback."""
    splits = {}
    params = None
    readback_errors = (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired)
    if psfgen is not None:
        from openmm import app
        from openmm.app.internal.charmm.exceptions import CharmmError

        readback_errors += (CharmmError,)
        params = app.CharmmParameterSet(str(ff.topology.source), str(ff.parameters.source))
    for case in cases:
        row = splits.setdefault(case.split, {
            "molecules": 0, "atoms": 0, "parameterized": 0, "stream_written": 0,
            "atoms_exported": 0, "errors": 0,
        })
        row["molecules"] += 1
        row["atoms"] += len(case.types)
        if psfgen is not None:
            row.setdefault("finite_energy", 0)
            row.setdefault("atoms_in_finite_systems", 0)
            row.setdefault("readback_errors", {})
        try:
            with rdBase.BlockLogs():
                result = parameterize_v5(case.mol, ff)
            if not result.is_complete or len(result.atoms) != len(case.types):
                continue
            row["parameterized"] += 1
            write_v5_stream(result, stream, residue_name="LIG")
            row["stream_written"] += 1
            row["atoms_exported"] += len(case.types)
        except (AutoCGENFFError, OSError):
            row["errors"] += 1
            continue
        if psfgen is not None:
            try:
                ok = _readback(case, result, stream, psfgen, params)
                category = "PSF, coordinates or energy mismatch"
            except readback_errors as exc:
                ok = False
                category = type(exc).__name__
            if ok:
                row["finite_energy"] += 1
                row["atoms_in_finite_systems"] += len(case.types)
            else:
                errors = row["readback_errors"]
                errors[category] = errors.get(category, 0) + 1
    return dict(sorted(splits.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ligandff", type=Path, required=True, help="pinned LigandFF checkout")
    ap.add_argument("--force-field", type=Path, default=REPO / "force_fields")
    ap.add_argument("--output", type=Path, required=True, help="aggregate JSON result path")
    lock = ap.add_mutually_exclusive_group(required=True)
    lock.add_argument("--write-manifest", type=Path, help="record initial inputs in a manifest")
    lock.add_argument("--manifest", type=Path, help="verify inputs against an existing manifest")
    ap.add_argument("--work", type=Path, default=Path("benchmark_local/work"), help="scratch dir")
    ap.add_argument("--psfgen", type=Path, help="also check PSF and finite energy with OpenMM")
    ap.add_argument("--sets", default="B0,B1,B2,B3,B4,QT,QF,QI,OUT")
    args = ap.parse_args()

    ligandff = args.ligandff.resolve()
    head = _ligandff_head(ligandff)
    if head != LIGANDFF_COMMIT:
        ap.error(f"expected LigandFF checkout at {LIGANDFF_COMMIT}, got {head}")
    wanted = {s.strip().upper() for s in args.sets.split(",") if s.strip()}
    if not wanted or wanted - SETS:
        ap.error(f"unknown or empty benchmark selection: {args.sets}")
    sys.path.insert(0, str(ligandff / "src"))
    from ligandff.benchmarks import b0, b1, b2, b3  # noqa: E402
    from ligandff.charges import IncrementTable  # noqa: E402

    manifest = _manifest(ligandff, args.force_field.resolve())
    if args.manifest:
        expected = json.loads(args.manifest.read_text(encoding="utf-8"))
        for key in ("schema", "input_checksums", "ligandff_commit", "rdkit_version"):
            if manifest[key] != expected[key]:
                ap.error(f"benchmark inputs changed: {key}")
    rtf = args.force_field / "top_all36_cgenff.rtf"
    ff = V5ForceField.from_directory(args.force_field)
    args.work.mkdir(parents=True, exist_ok=True)
    report = {"force_field": str(args.force_field), "ligandff_commit": head,
              "rdkit_version": rdBase.rdkitVersion}

    b1_cases = b2_cases = b3_cases = []
    if wanted & {"B1", "QF", "QI", "OUT"}:
        b1_cases, report["B1_build"] = b1.build(rtf)
        report["B1_build"]["note"] = "full 226-case test split incl. 23001B"
    if wanted & {"B2", "QF", "QI", "OUT"}:
        b2_cases, report["B2_build"] = b2.build(rtf)
    if "B3" in wanted or "OUT" in wanted:
        b3_cases, b3_excluded = b3.build()
        report["B3_build"] = {"cases": len(b3_cases), "excluded": b3_excluded}

    if "B1" in wanted:
        report["B1"] = {"typing": score_typing(b1_cases), "charges": score_charges(b1_cases, ff)}
    if "B2" in wanted:
        report["B2"] = {"typing": score_typing(b2_cases), "charges": score_charges(b2_cases, ff)}
    if "B3" in wanted:
        report["B3"] = {"typing": score_typing(b3_cases), "charges": score_charges(b3_cases, ff)}
    if "B0" in wanted:
        report["B0"] = score_b0(b0.load(ligandff / "tests" / "fixtures" / "si_v5_reference.json"), ff)
    if "QT" in wanted:
        report["QT"] = score_qt(rtf, ff)
    if "QF" in wanted:
        inc = IncrementTable.from_prm(args.force_field / "par_all36_cgenff_increments.prm")
        report["QF"] = {"B1": score_qf(b1_cases, inc), "B2": score_qf(b2_cases, inc)}
    if "QI" in wanted:
        report["QI"] = {"B1": score_qi(b1_cases), "B2": score_qi(b2_cases)}
    if "B4" in wanted:
        report["B4"] = score_b4(
            args.force_field / "par_all36_cgenff.prm",
            ligandff / "benchmarks" / "b4_annotations.split.json", ff, args.work,
        )
    if "OUT" in wanted:
        # Only the benchmark uses a fixed snapshot of the base files. Still
        # parse every exported fragment and verify the base hashes again below.
        def topology(path: Path):
            return ff.topology if path == ff.topology.source else read_topology(path)

        def parameters(path: Path):
            return ff.parameters if path == ff.parameters.source else read_parameters(path)

        with (
            TemporaryDirectory(dir=args.work.resolve()) as tmp,
            patch("autocgenff.v5_stream.read_topology", topology),
            patch("autocgenff.v5_stream.read_parameters", parameters),
        ):
            stream = Path(tmp) / "ligand.str"
            psfgen = args.psfgen.resolve() if args.psfgen else None
            report["OUT"] = {
                name: score_out(cases, stream, ff, psfgen)
                for name, cases in (("B1", b1_cases), ("B2", b2_cases), ("B3", b3_cases))
            }
        report["OUT"]["energy_readback"] = "psfgen and OpenMM" if args.psfgen else "not requested"
    if manifest["input_checksums"] != _manifest(ligandff, args.force_field.resolve())["input_checksums"]:
        raise ValueError("Benchmark inputs changed while scoring")
    report["generated_by"] = "tools/benchmark_v5_local.py"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            json.dumps(manifest, indent=1), encoding="utf-8"
        )
    print(json.dumps({"wrote": str(args.output)}, indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
