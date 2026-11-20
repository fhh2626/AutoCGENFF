"""Score one pre-isolated B1 split without building, repairing or typing the other."""

import argparse
import codecs
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory

from rdkit import Chem, rdBase

from autocgenff.errors import AutoCGENFFError
from autocgenff.v5 import type_v5

from .b1_structure import StructureInput, prepare_structure

TRAIN_RTF_SHA256 = "14a98450ac574bdf162300e9ab407c463022fe0f668b644ec51e1786990e32bd"
TRAIN_MANIFEST_SHA256 = "6536b819b6b949539390a44a2d4fe54d9f13845fdf7443ea5131524cf0ac1994"
TRAIN_COUNTS = (904, 22062, 65)


@dataclass(frozen=True, slots=True)
class FrozenEvaluationInput:
    """Parent-verified RTF checksum and path to the pinned train manifest."""

    rtf_sha256: str
    train_manifest: Path


@dataclass(frozen=True, slots=True)
class _VerifiedInputs:
    rtf_bytes: bytes
    manifest_bytes: bytes
    metadata: dict[str, StructureInput]


def _structure_inputs(rtf_bytes: bytes) -> dict[str, StructureInput]:
    """Read structure transport metadata, independent of RTF type/charge columns."""
    result = {}
    current = None
    with StringIO(rtf_bytes.decode("utf-8"), newline=None) as stream:
        for number, line in enumerate(stream, 1):
            words = line.partition("!")[0].split()
            if not words:
                continue
            if words[0] == "END":
                break
            if words[0] in {"RESI", "PRES"}:
                if words[0] == "PRES":
                    raise ValueError("split RTF must contain residues only, not patches")
                if len(words) < 3:
                    raise ValueError(f"RTF line {number}: RESI requires a name and net charge")
                current = words[1]
                charge = float(words[2])
                if current in result or not charge.is_integer():
                    raise ValueError(f"duplicate residue or non-integral net charge: {current}")
                result[current] = StructureInput(int(charge))
            elif current is not None and words[0].upper() in {"DOUB", "DOUBLE", "TRIP", "TRIPLE"}:
                # B1 aligns only connectivity and does not expose the RTF-name-to-index map.
                raise ValueError(f"{current}: explicit RTF bond-order cards are not supported")
    return result


def _manifest_names(splits: object, split: str, count: int) -> set[str]:
    if not isinstance(splits, dict) or set(splits) != {split} or not isinstance(splits[split], list):
        raise ValueError(f"manifest must contain only the requested split ({split})")
    names = splits[split]
    if (len(names) != count or any(not isinstance(name, str) for name in names)
        or len(set(names)) != count):
        raise ValueError("invalid or unexpected split manifest membership count")
    return set(names)


def _verify_inputs(
    rtf: Path, manifest: Path, split: str, expected: tuple[int, int, int],
    evaluation: FrozenEvaluationInput | None,
) -> _VerifiedInputs:
    # Pinned LigandFF opens RTFs with the interpreter's default text encoding.
    # It must tokenize the same verified bytes as this tool's UTF-8 preflight.
    with TextIOWrapper(BytesIO()) as default_text:
        if codecs.lookup(default_text.encoding).name != "utf-8":
            raise ValueError("B1 verification requires UTF-8 text I/O; run Python with -X utf8")
    molecules, atoms, repairs = expected
    if molecules <= 0 or atoms < molecules or not 0 <= repairs <= molecules:
        raise ValueError("nonpositive or inconsistent expected split counts")
    if split not in {"train", "test"} or (split == "test") != (evaluation is not None):
        raise ValueError("test scoring requires independently verified evaluation inputs")
    manifest_bytes = manifest.read_bytes()
    names = _manifest_names(json.loads(manifest_bytes.decode("utf-8")), split, molecules)
    if evaluation is None:
        expected_hash = TRAIN_RTF_SHA256
    else:
        train_bytes = evaluation.train_manifest.read_bytes()
        if hashlib.sha256(train_bytes).hexdigest() != TRAIN_MANIFEST_SHA256:
            raise ValueError("expected the pinned train manifest for the disjointness check")
        train_names = _manifest_names(json.loads(train_bytes), "train", TRAIN_COUNTS[0])
        if names & train_names:
            raise ValueError("evaluation manifest overlaps verified train membership")
        expected_hash = evaluation.rtf_sha256.lower()
    rtf_bytes = rtf.read_bytes()
    if hashlib.sha256(rtf_bytes).hexdigest() != expected_hash:
        raise ValueError(f"RTF SHA-256 does not match the verified {split}-only RTF")
    metadata = _structure_inputs(rtf_bytes)
    if metadata.keys() != names:
        raise ValueError("RTF residues do not match the split-only manifest; refusing to build")
    return _VerifiedInputs(rtf_bytes, manifest_bytes, metadata)


def _indexed_connectivity(mol: Chem.Mol) -> tuple[tuple[int, ...], set[tuple[int, int]]]:
    elements = tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
    edges = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.add((min(i, j), max(i, j)))
    return elements, edges


def score_split(
    rtf: Path, manifest: Path, split: str, expected: tuple[int, int, int], *, details: bool = False,
    evaluation: FrozenEvaluationInput | None = None,
) -> dict:
    """Score strict indexed equality; reject inconsistent split inputs with ValueError."""
    verified = _verify_inputs(rtf, manifest, split, expected, evaluation)
    # Optional development dependency, only used after preflight accepts both files.
    from ligandff.benchmarks import b1

    # The pinned API accepts paths only. Pass private copies of the verified bytes,
    # not caller paths that may now refer to different contents.
    with TemporaryDirectory(prefix="autocgenff-b1-") as directory:
        snapshot_rtf = Path(directory) / "split.rtf"
        snapshot_manifest = Path(directory) / "split.json"
        snapshot_rtf.write_bytes(verified.rtf_bytes)
        snapshot_manifest.write_bytes(verified.manifest_bytes)
        cases, build = b1.build(snapshot_rtf, manifest=snapshot_manifest)
    actual = (build["molecules"], build["atoms"], build["repaired_radicals"])
    if actual != expected or build["skipped"]:
        raise ValueError(f"B1 alignment mismatch: expected {expected}, got {build}")
    output = {}
    for protocol in ("pinned-b1", "structure-corrected"):
        counts = Counter(molecules=actual[0], atoms=actual[1], exact_molecules=0, exact_atoms=0,
                         rejected_molecules=0, incomplete_molecules=0)
        sources = Counter()
        misses = {}
        for case in cases:
            molecule = case.mol
            if protocol == "structure-corrected":
                raw = Chem.MolFromSmiles(case.smiles)
                if raw is None:
                    raise ValueError(f"B1 SMILES could not be parsed for {case.name}")
                result = prepare_structure(Chem.AddHs(raw), case.mol, verified.metadata[case.name])
                sources[result.source] += 1
                molecule = result.molecule
                if molecule is None:
                    counts["rejected_molecules"] += 1
                    misses[case.name] = result.source
                    continue
                if _indexed_connectivity(molecule) != _indexed_connectivity(case.mol):
                    raise ValueError(f"structure repair changed B1 indexed connectivity: {case.name}")
            if molecule.GetNumAtoms() != len(case.types):
                raise ValueError(f"B1 atom count changed for {case.name}")
            try:
                typed = type_v5(molecule)
            except AutoCGENFFError as exc:
                counts["rejected_molecules"] += 1
                misses[case.name] = str(exc)
                continue
            counts["incomplete_molecules"] += not typed.is_complete
            mismatches = [
                (i, predicted, reference)
                for i, (predicted, reference) in enumerate(zip(typed.types, case.types, strict=True))
                if predicted != reference
            ]
            counts["exact_atoms"] += len(case.types) - len(mismatches)
            counts["exact_molecules"] += not mismatches
            if details and mismatches:
                misses[case.name] = [
                    {
                        "index": i,
                        "predicted": predicted,
                        "reference": reference,
                        "neighbors": sorted(
                            (a.GetIdx(), a.GetSymbol())
                            for a in molecule.GetAtomWithIdx(i).GetNeighbors()
                        ),
                    }
                    for i, predicted, reference in mismatches
                ]
        output[protocol] = {"counts": counts}
        if sources:
            output[protocol]["structure_sources"] = sources
        if details:
            output[protocol]["misses"] = misses
    return {"split": split, "alignment": build, "rdkit": rdBase.rdkitVersion, "scores": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rtf", type=Path)
    parser.add_argument("--manifest", type=Path, required=True, help="pre-isolated split manifest")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--require-exact", action="store_true", help="gate structure-corrected score")
    parser.add_argument("--details", action="store_true", help="emit indexed mismatches to stdout")
    parser.add_argument("--expected-test", nargs=3, type=int, metavar=("MOLECULES", "ATOMS", "REPAIRS"))
    parser.add_argument("--expected-test-sha256", help="independently verified evaluation RTF SHA-256")
    parser.add_argument("--train-manifest", type=Path, help="pinned train manifest for disjointness check")
    args = parser.parse_args()
    evaluation = None
    expected = TRAIN_COUNTS
    evaluation_args = (args.expected_test, args.expected_test_sha256, args.train_manifest)
    if args.split == "test":
        if not all(evaluation_args):
            parser.error("--split test requires --expected-test, --expected-test-sha256 and --train-manifest")
        expected = tuple(args.expected_test)
        evaluation = FrozenEvaluationInput(args.expected_test_sha256, args.train_manifest)
    elif any(evaluation_args):
        parser.error("evaluation options may only be used with --split test")
    try:
        output = score_split(
            args.rtf, args.manifest, args.split, expected, details=args.details, evaluation=evaluation,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, indent=2, sort_keys=True))
    corrected = output["scores"]["structure-corrected"]["counts"]
    return int(args.require_exact and (
        (corrected["exact_molecules"], corrected["exact_atoms"]) != expected[:2]
        or corrected["rejected_molecules"] or corrected["incomplete_molecules"]
    ))


if __name__ == "__main__":
    raise SystemExit(main())
