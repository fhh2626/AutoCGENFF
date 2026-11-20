# B1 train-only typing check

Run from the source checkout with RDKit and the [pinned LigandFF checkout](https://github.com/carlosbaiz/ligandff/tree/860b17da7e982ef7ab62d993a0e24fda3958bbb6). Set `LIGANDFF_CHECKOUT` to that checkout and `B1_TRAIN_DIR` to the directory containing the pre-isolated `b1-train.rtf` and `b1-train.json`.

```sh
PYTHONPATH="src:$LIGANDFF_CHECKOUT/src" python -X utf8 -m tools.score_v5_b1_split \
  "$B1_TRAIN_DIR/b1-train.rtf" --manifest "$B1_TRAIN_DIR/b1-train.json" \
  --split train --require-exact
```

Use the train-only RTF and its matching train-only manifest, never the full topology or a combined manifest. The scorer verifies the inputs before alignment. It reports pinned B1 and structure-corrected inputs separately. Structure correction may use the residue net charge; the atom typer does not receive reference labels. Raw SMILES, pinned B1 inputs and structure-corrected inputs are distinct protocols.

`--require-exact` returns 1 unless every indexed label in the structure-corrected split matches, including explicit H and alternating type digits. Add `--details` to print mismatches to standard output. Do not evaluate the fixed split until the strict train gate passes and the rules are frozen. Never use fixed-split results to tune the rules; this split has previously been inspected and is not a blind evaluation.
