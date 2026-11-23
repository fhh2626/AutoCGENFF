# Local-v5 ligand typing

`autocgenff type-v5` and `autocgenff.v5.type_v5` apply the installed `cgenff_v5.rules` to a SMILES or a path to one SDF or MOL2 molecule. The Python API also accepts an RDKit `Chem.Mol`. Provide the intended protonation state and bond orders. Existing atom indices are preserved and missing hydrogens are appended. Typing does not require a reference residue, force-field files or a network connection.

It is an independent implementation for CGenFF 5.0, not the official CGenFF program.

## Reading a typing result

- `atom_type` is the first matching type in the decision tree; `trace` records the rule path. An unmatched atom has `atom_type=None` and an error. A complete typing means every atom received a label, not that parameters or charges are available.
- `formal_charge` describes the normalized working graph. Unambiguous charge-separated amide and ester drawings are normalized to the neutral carbonyl contributor without moving H or changing component charge. `rule_charge` is a separate seed for the charge-increment model. Neither value is a partial atomic charge.
- A local typing extension recognizes the neutral five-membered-ring imine role (`NG2R50`) next to guanidinium. It requires a retained ring C=N bond and an amino nitrogen bound only to H or saturated carbon outside the guanidinium group that could carry the positive charge in an equivalent Lewis drawing. The override is skipped if the retained imine carbon has a competing amino donor. The override itself changes only the imine type, not the input bonds or the charge seed assignment; other atoms' types can still depend on the drawing.
- A local fused-ring rule assigns `NG2R52` to a five-membered-ring fusion N when an in-ring neighbour is double-bonded to an in-ring cation. A distant or saturated ammonium cannot trigger this rule. The aromatic and all-sp² five-membered-ring cation rules still apply independently.
- Local sulfonamide rules distinguish an N-H single-bonded to a carbonyl carbon and an oxidized sulfur (`NG2S1`) from an aryl sulfonamide N-H without a charged oxygen arm (`NG311`). An N-sulfonyl lactam carbonyl can keep the plain amide type `CG2O1`. That override excludes four- and five-membered rings and carbonyls with two single-bonded N, O or S neighbours.
- The carbonyl-dependent `CG2R62` branch checks for a C=O-bearing carbon in the same aromatic six-membered ring. A ring thione or N-sulfonyl lactam carbonyl cannot trigger this branch. Adjacent ring hydrogens can still take `HGR62` through the separate substituent rule. The cationic `CG2R62` branch is unchanged.
- `improper_center` marks an atom for improper parameter lookup. The typer does not create a virtual site or assign an improper force constant.
- Ring perception uses integer Kekulé bond orders. Tautomers, protonation states and equivalent atom orderings can produce different labels, including the digits of alternating types. There is no global resonance search or general bond-order repair.
- Neutral water, including isotopic or detached water, belongs to `toppar_water_ions.str`. Radicals and unsupported structures are rejected. Rare-element rules cover specific bonding environments only.

## Local-v5 parameterization

Use `autocgenff.parameterize_v5(molecule)` or pass a reusable `V5ForceField.from_directory()`. Both default to the bundled CGenFF 5.0 files in `autocgenff/force_fields/`. Pass an explicit `directory` (with compatible `top_all36_cgenff.rtf`, `par_all36_cgenff.prm`, and `par_all36_cgenff_increments.prm`) or `increments_path=` to override them.

Inspect `is_complete`, `issues`, `terms`, `lone_pairs` and `charge_diagnostics` before export. Charge seeds must sum to the formal charge of each connected component. The charges in `charge_diagnostics` are the real-atom totals from increments. Atom `partial_charge` values are the exported site charges after any transfer to a virtual site. Unresolved paths leave charges unavailable.

A torsion path with an sp carbon (`CG1N1`, `CG1T1` or `CG1T2`) in a central position has an undefined dihedral angle. If the increment file has no row for such a path, it transfers no charge (`linear_not_applicable`). This follows the native file, where every annotated row of this kind is `N/A`. The rule applies only if the file has at least one such row and all of them are annotated `N/A`. An exact row still takes precedence.

Mechanical parameters and charge increments are matched independently. The Part II directional substitution model (`observed_v5_penalty_model`) combines the published `NG3` hierarchy and bond groups with single- and iteratively solved multi-substitution observations from the supplied PRM, zero-cost alternating-digit twins, and scaled relation fallbacks. It does not substitute LJ sites or invent missing `LPH` entries. Aromatic halogens (`CLGR1`, `BRGR1`, `IGR1`) receive collinear `LPH` lone pairs (`+0.05 e` at `1.64`, `1.89` and `2.20` Å with `SCALE 0.0`), subtracting `0.05 e` from the host halogen site charge while conserving residue charge.

Equal-parameter improper ties can resolve using native topology evidence or an energy-equivalent swap of the two middle atoms. Competing force constants or phases remain unresolved. Two-coordinate cumulene centers, including carbodiimides, are not marked for three-neighbor impropers. Unsupported isotope masses prevent completion.

## Stream export

See the [CLI and Python examples](../README.md#parameterize-a-ligand) or run `autocgenff assign-v5 --help`. `write_v5_stream(assignment, path, residue_name="LIG")` writes a single connected residue, including any aromatic halogen `LONEPAIR COLINEAR ... SCALE 0.0` virtual sites. Atom names must be unique without regard to case and representable in CHARMM. Use coordinates with matching names and atom order; the stream contains neither coordinates nor a PSF.

Load the compatible base RTF and PRM before the stream, including any system force fields needed for external NBFIX partners. Keep angle and dihedral autogeneration enabled. The writer emits only missing bonded parameters assigned by analogy; exact and wildcard matches, LJ, 1–4 and NBFIX parameters stay in the base files. Check for collisions with other streams in the same system. Missing terms, unresolved charges or lone pairs, and omitted linear torsions can block export. A readable stream does not establish simulation readiness.

## Validation commands

Run the typing checks with `python -m pytest -q tests/test_v5.py tests/test_v5_reference.py`, and the parameter and export checks with `python -m pytest -q tests/test_v5_assignment.py tests/test_v5_engine.py`. Both use the bundled CGenFF 5.0 files by default; set `AUTOCGENFF_V5_DIR` and `AUTOCGENFF_V5_INCREMENTS` to override them, and set `AUTOCGENFF_PSFGEN` to a `psfgen` executable for optional PSF checks.
