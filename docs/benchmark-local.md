# Local v5 benchmark suite

`tools/benchmark_v5_local.py` scores AutoCGENFF against the benchmark inputs from LigandFF `860b17d`. The runner writes aggregate results only for the sealed test splits. LigandFF supplies case builders, split membership and reference fixtures. It does not predict AutoCGENFF's atom types, partial charges, or export results.

## Prepare the local inputs

The working directory `benchmark_local/` is ignored by Git and excluded from packages. Keep the three native files in `force_fields/`. Use a clean LigandFF checkout at `860b17d` or export the required paths from that commit:

```sh
mkdir -p benchmark_local/inputs/ligandff
git -C /path/to/LigandFF archive 860b17d src benchmarks tests/fixtures \
  | tar -xf - -C benchmark_local/inputs/ligandff
git -C /path/to/LigandFF rev-parse 860b17d \
  > benchmark_local/inputs/ligandff/.source-commit
```

The first run uses `--write-manifest benchmark_local/manifest.json` to lock the RDKit version and SHA-256 hashes of the RTF, PRMs, LigandFF builders, split manifests and fixtures. Subsequent runs use `--manifest benchmark_local/manifest.json` to verify the same inputs. Builders reconstruct the cases without copying the native RTF and PRMs or maintaining duplicate labels.

```sh
micromamba activate autocgenff
python tools/benchmark_v5_local.py \
  --ligandff benchmark_local/inputs/ligandff \
  --manifest benchmark_local/manifest.json \
  --output benchmark_local/revised.json
```

`--sets B0,B1,B2,B3,B4,QT,QF,QI,OUT` controls which checks run. To include independent engine verification in `OUT`, install OpenMM in the benchmark Python environment and pass `--psfgen /path/to/psfgen`. A Windows executable under WSL requires `--work` on a Windows-accessible path. Without `--psfgen`, `OUT` reports export coverage only. The runner uses a verified, fixed snapshot of the parsed base files for repeated export checks, so it does not reread the same large RTF and PRM for every ligand. It still uses the normal stream writer and checks the input hashes again at the end.

## What each score measures

| Check | Source and unit |
| --- | --- |
| B0 | Three Table S10 molecules and 55 atoms. The replay applies native increments using the printed analogy source and penalty where needed. Charges and charge penalties must agree to the printed 0.001 precision. This checks the charge calculation with fixed inputs, not the prediction of those inputs. |
| B1 | 904 training and 226 test molecules. Native RTF types and charges, with input structures built from the SMILES comments by the pinned builder. This includes `23001B`. It is distinct from the separate 225-molecule filtered protocol in [B1 training](b1-train-protocol.md). |
| B2 | 728 training and 182 test molecules. Legacy RTF model compounds without SMILES. The builder infers bond orders from connectivity and net charge, so some reference comparisons may use an uncertain structure. |
| B3 | 34 development molecules reconstructed from CHARMM-GUI GROMACS topologies. The reference types and real-atom charges come from those topologies. |
| B4 | 5,787 training and 1,447 test parameter annotations. Report counts of source matches and penalty errors of at most 0.5, and their intersection. Fits the CGenFF Part II directional cost model (`observed_v5_penalty_model`) using training annotations only. The candidate pool excludes the target and, for batch cases, the annotated parameter batch. There is no atom or molecule accuracy for this check. |
| QT | Native increments evaluated on reference atom types with a per-atom integer seed inferred from the reference charges. Only residues with applicable reference increments and no lone pairs, external bonds or fewer than two atoms qualify. Reports residuals after subtracting the inferred integer seed. This measures fit to existing charges as opposed to predicting seeds or types. |
| QF | Under the pinned protocol, compare `rule_charge` against the per-atom integer seed inferred using reference types and increments. Eligibility requires canonically correct types, complete numeric reference increments and no lone pairs. |
| QI | On canonically correctly typed B1 and B2 molecules, compare improper center membership with RTF `IMPR` declarations. This does not compare the order or energy of a specific improper. |
| OUT | B1, B2 and B3 parameterization, checked stream export and optional PSF and OpenMM readback. Counts real atoms belonging to passing molecules. Finite energy on an embedded starting geometry is a compatibility check, not an accuracy measure or evidence of simulation readiness. |

Typing accuracy is strict, atom-index aligned equality. The JSON also reports a separate canonical score that folds the alternating type-number conventions used by LigandFF. Partial-charge accuracy uses absolute error strictly below 0.1 e. A molecule is correct only if every eligible atom passes. Unresolved charges count as failures. Reference cases with explicit lone pairs are excluded from the real-atom charge comparison because the charges depend on those sites. Median per-molecule charge RMSE is over resolved molecules only. QF and QI denominators can grow when atom typing improves.

## Changes evaluated here

- The acyclic monoamide anion uses `NG2D1` with a seed of −1 on N whether the charge is drawn on N or in the corresponding imidate resonance form on O. Imides and unrelated anionic nitrogen are excluded. The draws keep their input bond and formal charge representation. The reference model compounds now yield partial charges, although the original increment table still misses the 0.1 e threshold on some atoms.
- The terminal oxygen of a sulfenate uses `OG312` with a seed of −1. Neutral sulfanyl peroxide and S-nitrosothiol oxygens get their reference oxygen types, and terminal S–H keeps the thiol type in a hydrodisulfide. Sulfate and phosphate rules remain separate.
- A two-coordinate carbon with two double bonds does not receive a three-neighbor improper. This includes isothiocyanates and carbodiimides, and does not infer new mechanical parameters.
- Improper ties resolve through topology evidence or energy-equivalent swaps when the parameter values agree. Distinct force constants or phases remain unresolved even when one ordering puts equivalent atoms in the two middle positions.
- `observed_v5_penalty_model` adds the CGenFF Part II `NG3` hierarchy, bond groups, single- and multi-substitution PRM observations, alternating-digit twins and scaled relation fallbacks, backed by exact `CandidateIndex` pruning.
- Aromatic halogens (`CLGR1`, `BRGR1`, `IGR1`) generate collinear `LPH` lone pairs (`+0.05 e`, `SCALE 0.0`) subtracted from the host halogen site charge, while aliphatic halogens (`CLGA1`, `BRGA1`) remain real-atom sites.

No increment values or reference labels were changed. Unsupported ionic families, genuinely conflicting improper orders and omitted linear torsions remain limitations. The comparison used an unmodified source snapshot of `d7993a8` for the baseline and the same locked data for the revised runs. The baseline report is `benchmark_local/baseline.json`. The final comparison is `benchmark_local/final-review.json`. The separate 98-structure stress set has no reference atom labels or charges. Its results measure coverage only.
