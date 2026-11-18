# AutoCGENFF

An independent Python implementation of selected methods from CGenFF [Part I](https://doi.org/10.1021/ci300363c) (perception and atom typing) and [Part II](https://doi.org/10.1021/ci3003649) (parameter analogy and charge increments). **It does not reproduce the official CGenFF atom-typing decisions, penalties, charge fit, or scores.** The full official rules, matrices, and fit inputs are not published in those papers. For supported ligands it can write a local additive CHARMM stream and a JSON audit. Successful export checks coverage and representation, **not** simulation readiness.

## Install and run

Python 3.11+; Python 3.12 is used in the reference environment. RDKit handles chemistry and SciPy/NumPy provide staged sparse least squares.

```sh
micromamba create -f environment.yml
micromamba activate autocgenff
python -m pip install --no-deps -e .
python -m pytest
ruff check .
python -m build --no-isolation
```

Or install with `python -m pip install '.[dev]'` in a suitable Python environment. Supply your own compatible additive CHARMM PRM/RTF files. The v5 source force fields, paper PDFs, and local coding guide are not distributed. The tests use original fixtures; optional v5 tests run only when the local files are present.

### Parameterize a ligand in one offline invocation

Supply the unmodified `top_all36_cgenff.rtf` and `par_all36_cgenff.prm` in one directory plus a compatible native `par_all36_cgenff_increments.prm`. No reference residue name, custom typing, JSON model, network connection or water topology is needed. For example, with `CGENFF_INCREMENTS` set to your local [v5 increment PRM](docs/local-v5.md#local-v5-parameterization):

```sh
autocgenff assign-v5 --smiles 'CC(C)(C)CO' --force-field force_fields \
  --increments "$CGENFF_INCREMENTS" --residue NPOL --stream neopentyl.str > neopentyl.json
```

This example assigns neopentyl alcohol directly from structure, including three occurrences of an inferred torsion. Use `--input ligand.mol2` or `--input ligand.sdf` instead of `--smiles` to use a file. Omit `--increments` if the conventional increment filename is beside the base PRM. Omit `--stream` for a JSON-only audit. `python -m autocgenff` also works. Use a distinct `.json` path for shell redirection: **do not redirect stdout to your stream or input file**.

Exit **0**: assignment complete relative to the local models (and stream written if requested). **1**: audit with unresolved terms/sites; no stream written. **2**: input or export error, JSON error on stderr. If assignment succeeded before export failed, stdout also contains the assignment and `stream_error`. Only use a stream from a successful invocation: a failed invocation preserves an older output file. Inspect typing, charge and parameter provenance even on success.

In CHARMM, first load the compatible base RTF/PRM (and your system's partner force fields), then `stream neopentyl.str`. The new residue here is `NPOL`. Do not replace the base files: exact, wildcard, LJ and NBFIX records are intentionally not repeated. Do not disable angle/dihedral autogeneration. The stream is not a coordinate file or a PSF. Use coordinates matching the audit's atom names and order and verify the generated system. See [format, independent engine checks and limits](docs/local-v5.md#stream-export).

### Offline atom typing only (no force-field files needed)

```sh
autocgenff type-v5 --smiles 'CC(=O)N'
# or: python -m autocgenff type-v5 --input ligand.sdf
```

```python
from autocgenff.v5 import type_v5

result = type_v5("O=C=O")
print(result.is_complete, result.types)  # True ('OG2D5', 'CG2O7', 'OG2D5')
print(result.atoms[0].trace)  # matching DSL categories, lines, jumps and leaf
```

`type-v5` executes the entire bundled **local-v5** decision tree, not the starter SMARTS below. It accepts only a structure and reports real atoms (including H); it does not assign partial charges, bonded parameters, impropers or virtual sites. CLI status 0 means every atom reached a type, **not** that those types are validated; 1 means at least one unmatched atom (`null` with a reason); 2 means invalid/unsupported input or failed perception. No runtime download or LigandFF install. See [profile, differences and exact-map validation](docs/local-v5.md).

### Local-v5 parameterization (no template or custom model)

```python
from pathlib import Path
from autocgenff import V5ForceField, parameterize_v5, write_v5_stream

# Here the native increments are beside the base files; load once for a batch.
ff = V5ForceField.from_directory(Path("force_fields"))
result = parameterize_v5("CC(=O)[O-]", ff)
print(result.is_complete, result.issues)
if result.is_complete:
    write_v5_stream(result, Path("acetate.str"), residue_name="MYAC")
```

Supply compatible `par_all36_cgenff.prm`, `top_all36_cgenff.rtf` and `par_all36_cgenff_increments.prm`. For increments kept separately, pass `increments_path=Path(...)` to the loader. A one-shot call also accepts `parameterize_v5(smiles, directory, increments_path=...)`. No downloads. See [native inputs, inventory, provenance and limits](docs/local-v5.md#local-v5-parameterization). **Neutral water, including detached and isotopic water, belongs to `toppar_water_ions.str`**, not the ligand typer. The explicit named template reader below remains available.

### Named local templates

```sh
autocgenff --prm force_fields/par_all36_cgenff.prm --rtf force_fields/top_all36_cgenff.rtf \
  --smiles CCO --template ETOH
autocgenff --prm force_fields/par_all36_cgenff.prm --rtf force_fields/top_all36_cgenff.rtf \
  --smiles c1ccccc1 --template BENZ --template-smiles c1ccccc1
autocgenff --prm force_fields/par_all36_cgenff.prm --rtf force_fields/top_all36_cgenff.rtf \
  --smiles 'C[NH3+]' --template MAMM --template-smiles 'C[NH3+]'
```

**You identify the residue.** This is not a database search or an automatic declaration of chemical identity. The code checks charge totals, explicit-H atom count, MASS elements, covalent connectivity and bond orders where encoded. Aromatic, charged, radical, isotopic, or stereochemically relevant targets require an independently verified `--template-smiles` identifying the *reference*, not simply a copy of an unknown query. The reference identity must match RDKit's isomeric SMILES for the query; RTF aromatic `BOND`/`DOUBLE` entries are then permitted on aromatic query edges. Enhanced stereochemistry groups and atropisomeric bond stereo are not supported in the query or the reference for named matching. Molecule preparation rejects query bonds and explicit allene stereochemistry instead of verifying a simplified structure. RTF does not encode atom-local formal charges or complete stereochemistry. Where correspondences differ in assigned charges, types, or dependencies, provide a vetted full `--atom-map` JSON from RTF atom names to zero-based explicit-H atom indices. Duplicate JSON names, including case variants, are rejected. The same topology and identity checks still apply. Residues requiring virtual sites, patches, unsupported directives or external atoms fail clearly. Missing `MASS` element declarations are not inferred from names or masses.

### Legacy SMARTS assignment and charge diagnostics

```sh
autocgenff --prm force_fields/par_all36_cgenff.prm --smiles CCO --no-impropers \
  --increments examples/ethanol-bond-increments.json
```

The included ethanol table is **illustrative synthetic data**, not a validated charge set. `--no-impropers` is a caller assertion, not inference. Without a template, `--impropers` takes a JSON array of ordered atom-index quartets. An absent declaration leaves a diagnostic issue. CLI returns **0** for complete diagnostic coverage, **1** for JSON with missing items, and **2** for invalid/unsupported input (JSON error on stderr). Successful coverage is not evidence of parameter accuracy. JSON records use the native PRM row order and units; a `null` charge is not zero.

### Python API

```python
from pathlib import Path
from autocgenff import AssignmentOptions, parameterize, read_parameters, read_topology

prm = read_parameters(Path("force_fields/par_all36_cgenff.prm"))
rtf = read_topology(Path("force_fields/top_all36_cgenff.rtf"))
result = parameterize("CCO", prm, AssignmentOptions(topology=rtf, template="ETOH"))
print(result.is_complete, [atom.partial_charge for atom in result.atoms])
```

`parameterize` and `prepare_molecule` accept SMILES strings with one nonblank line, paths to one MOL2/SDF molecule, or RDKit `Mol` objects. RDKit sanitizes and assigns aromaticity, rings, formal charge and stereo; this is **not** Part I's ring list, aromaticity rules, bond-order/resonance repair or programmable rule tree. Existing atoms retain indices; new H atoms are appended. Use the correct protonation, tautomer and bond orders. RDKit's MOL2 reader expects supported Tripos/Corina-style typing; MOL2 partial-charge values are not used as formal charges or force-field charges. Text input accepts plain SMILES only. CXSMILES annotations and trailing names are rejected to prevent silent information loss. SDF supports named data fields (`<name>`); number-only data headers are not supported by the RDKit reader.

In the legacy `parameterize` workflow (not `parameterize_v5` or `type-v5`), the starter SMARTS cover selected acyclic saturated hydrocarbons and alcohols, isolated acyclic hydrocarbon alkenes with H on both alkene carbons (including propene and both butene connectivities), benzene, and simple primary alkylammonium groups. Generic rules are heuristics limited to these environments, not an exhaustive chemical domain. Other environments (including halogens, conjugated systems, many rings and heterocycles) need an explicitly vetted template or user rules. Unknown atoms identify the failing indices. Use `TypingRule(name, smarts, atom_type, priority=0, target=0)` or `--rules` with a JSON array of those keyword objects. Highest priority wins; first rule wins ties, user rules precede built-ins on ties, and `target` is the zero-based atom position in the query. `--no-builtin-rules` disables built-ins.

## Parameter and charge behavior

- Standalone additive CHARMM PRM bond, angle (including Urey–Bradley columns), proper (all multiplicities), improper, LJ (including optional 1–4 columns), and NBFIX rows. Diagnostics retain source, line and native row types; `record_reversed` gives one orientation flag per row. The group-level `reversed` flag is `null` for a mixed-orientation family. Native LJ lookup prefers an exact type, then the last matching `*`, `%`, `#`, or `+` pattern in file order. NBFIX inventory lists applicable native rows for types present, including `X` and CHARMM `*`, `%`, `#`, `+` patterns; it is not a pair/exclusion list or a substitute for loading partner force fields. NBFIX pair lookup uses the last applicable row in file order. For torsions, exact matches take precedence and are not mixed with wildcard Fourier components. Exact improper matches retain all specified rows, including reversed rows. Proper wildcards are limited to `X-A-B-X` and use file order; the [CHARMM improper wildcard families](https://academiccharmm.org/documentation/version/c47b1/io/#ParamFiles) are tried in this order: `A-X-X-B`, `X-A-B-C`, `X-A-B-X`, `X-X-A-B`. File order breaks ties within each family. Improper declarations retain order (complete reversal allowed; no arbitrary permutations). Named matching uses graph-generated angles or propers when enabled by RTF `AUTO` and ignores explicit declarations for those kinds.
- The legacy API has no default parameter analogy or LJ substitution. Optional `PenaltyModel` specifies directional `(target, source)` substitution costs; each missing off-diagonal prohibits that substitution. Minimum score uses a 10× inner-atom weight, bonded vs nonbonded matrices, and optional pairwise `BondGroup` mismatch penalties. Improper analogy aligns the central atoms before scoring and only uses source rows whose center can be established from a matching typed improper declaration in the supplied RTF. Without an RTF or with absent, ambiguous, or conflicting center evidence, the candidate is skipped. This does not change exact or wildcard lookup. Group alternatives sharing one name are equivalent. Example `--penalties` JSON: `{"bonded":[["TARGET","SOURCE",1.5]],"nonbonded":[],"bond_groups":[{"name":"ring","members":["A","B"],"penalty":20}],"max_penalty":30}`. This illustrates the schema, not a recommended substitution. Scores are not official or calibrated.
- A `ChargeModel` or `--increments` JSON supplies an ordered type key and one transfer per adjacent edge; e.g. `{"max_order":2,"entries":[{"types":["A","B"],"values":[0.1]}]}` moves 0.1 e from A to B. Set `max_order` to 2 (bond), 3 (through angle), or 4 (through proper torsion, default). Reversing a path reverses and negates the increments. Palindromic paths accept only reversal-invariant vectors, e.g. `(x,-x)` for an angle (identical-type bonds necessarily have zero transfer). Analogy uses the nonbonded matrix for all atom positions and refuses orientation-dependent sources for palindromic targets. Above `torsion_cutoff` (finite and positive; default 50), selected torsion increments are zeroed and accrue a capped nonzero penalty. Charge penalty is the square root of the sum of `(|net transfer onto this atom from one path| + 0.05**6)**(1/3) × penalty**2` contributions. Missing entries (except the mathematical identical-type bond zero) leave all charges unset with path diagnostics; they are not assumed zero.
- Increments preserve the RDKit formal-charge total on each fragment. Use `formal_baseline`/`--formal-baseline` for caller-reviewed resonance-symmetric seed charges (same per-fragment sums); otherwise localized RDKit formal charges may yield asymmetry. The library does not redistribute formal charges automatically.
- `autocgenff.fitting.fit_increments` accepts vetted `TypedReference` objects with explicit-H molecule, type tuple, partial-charge tuple, and optional baseline. It uses sparse, ridge-restrained least squares in bond → angle → torsion stages on successive residuals, with palindromic paths fixed at zero in fitting. It returns a reusable `ChargeModel` and training-stage RMS errors. There is no official training-set extraction, rounding, conjugation or hydrogen-charge constraints, official analogy calibration, or automatic held-out validation; provide independently checked references and validate out of sample.

No RTF patch application, virtual-site generation, Drude, CMAP assignment, explicit exclusions, coordinate optimization or PSF writer. The stream writer is specific to `parameterize_v5`, not legacy `parameterize`. Keep the compatible original force fields and audit with the stream; do not treat `is_complete` as permission to run a simulation.
