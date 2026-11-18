"""Development-only local B1 evaluation; see docs/local-v5.md for pinned inputs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ligandff.benchmarks import b1
from rdkit import Chem, rdBase

from autocgenff.errors import AutoCGENFFError
from autocgenff.v5 import type_v5, v5_vocabulary


def score(rtf: Path) -> dict:
    """Use B1's fixed alignment/split; never pass reference labels or residue IDs to inference."""
    cases, build = b1.build(rtf)
    results = defaultdict(Counter)
    reference_labels = defaultdict(set)
    correct_labels = defaultdict(set)
    for case in cases:
        raw = Chem.AddHs(Chem.MolFromSmiles(case.smiles))
        # B1 can repair bond orders using the reference residue net charge. Count the
        # raw-SMILES result separately, without that privileged preprocessing.
        for protocol, molecule in (("raw", raw), ("b1", case.mol)):
            key = f"{protocol}/{case.split}"
            counts = results[key]
            reference_labels[key].update(case.types)
            counts["molecules"] += 1
            counts["atoms"] += len(case.types)
            if molecule.GetNumAtoms() != len(case.types):
                raise ValueError(f"B1 atom count changed for {case.name}")
            try:
                typed = type_v5(molecule)
            except AutoCGENFFError:
                counts["rejected_molecules"] += 1
                continue
            counts["incomplete_molecules"] += not typed.is_complete
            correct = [r for p, r in zip(typed.types, case.types, strict=True) if p == r]
            hits = len(correct)
            correct_labels[key].update(correct)
            counts["exact_atoms"] += hits
            counts["exact_molecules"] += hits == len(case.types)
    for key, counts in results.items():
        counts["distinct_reference_labels"] = len(reference_labels[key])
        counts["labels_with_at_least_one_correct_atom"] = len(correct_labels[key])
    return {
        "alignment": build,
        "scores": results,
        "profile_vocabulary": len(v5_vocabulary()),
        "rdkit": rdBase.rdkitVersion,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rtf", type=Path)
    print(json.dumps(score(parser.parse_args().rtf), indent=2, sort_keys=True))
