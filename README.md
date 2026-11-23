# AutoCGENFF

**Offline ligand parameterization for CGenFF 5.0, developed for BFEE-Docking.**

AutoCGENFF is an independent Python implementation of ligand atom typing, partial-charge assignment, and bonded parameter assignment for the CHARMM General Force Field (CGenFF) 5.0. It was developed to provide an open-source ligand preparation tool for [BFEE-Docking](https://github.com/fhh2626/BFEE-Docking). Given a small molecule and compatible CGenFF 5.0 force-field files, it produces an additive CHARMM stream and a JSON report documenting the source and penalty of each assignment. Docking and binding free-energy calculations are handled by downstream workflows.

The project was motivated by [LigandFF](https://github.com/carlosbaiz/ligandff) ([ChemRxiv preprint](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15011711/v1)). AutoCGENFF targets the CGenFF 5.0 data format and is independent of the official CGenFF program, so its atom types, partial charges, parameter choices, and penalties can differ from official CGenFF output.

## How to cite

If you use AutoCGENFF, please cite:

Sun, X., Li, Z., Shao, X., Cai, W., Li, Z., and Fu, H. "BFEE-Docking: A User-Friendly and Customizable End-to-End Tool from High-Throughput Virtual Screening to Binding Free-Energy Calculations." *Journal of Chemical Theory and Computation* **2026**, *22*, 6976-6985. [doi:10.1021/acs.jctc.6c00830](https://doi.org/10.1021/acs.jctc.6c00830).

## What AutoCGENFF does

| Task | Description |
| --- | --- |
| **Atom typing** | Assigns CGenFF 5.0 atom types to all explicit atoms, including hydrogens, with a rule trace for each atom. Works without external force-field files. |
| **Charge assignment** | Applies CGenFF charge increments to rule-based charge seeds and reports partial charges, per-atom charge penalties, and increment sources. |
| **Bonded parameter assignment** | Uses exact and wildcard records from the parameter file where available, searches for bonded analogs for supported missing terms, and reports each source record and penalty. |
| **CHARMM export** | Writes a single ligand residue, aromatic halogen collinear lone pairs when required, and only the bonded parameter additions needed by that residue, for use alongside the base RTF and PRM files. |

Both the CLI and the Python API run completely offline. Supported inputs are a plain SMILES string or an SDF or MOL2 file containing a single molecule, and the Python API also accepts an RDKit `Mol` object. Existing atom indices are preserved and missing hydrogens are appended. Provide structures with the intended protonation state, tautomer, and bond orders. PDB files are not supported as ligand input.

## Installation

Install from PyPI (Python 3.11 or newer):

```sh
pip install autocgenff
```

To install directly from the `master` branch:

```sh
pip install git+https://github.com/fhh2626/AutoCGENFF.git@master
```

The `master` branch can always be treated as the latest version, and GitHub Releases will not be updated.

## Quick start

### Atom typing only

```sh
autocgenff type-v5 --smiles 'CC(=O)N'
```

`type-v5` reports atom types, charge seeds, and rule traces without calculating partial charges or bonded parameters. Pass `--input ligand.sdf` or `--input ligand.mol2` instead of `--smiles` to read from a file. You can also invoke the CLI with `python -m autocgenff`.

### Parameterize a ligand

Assign neopentyl alcohol from its SMILES, write an additive CHARMM stream, and save the JSON report to a separate file:

```sh
autocgenff assign-v5 --smiles 'CC(C)(C)CO' --force-field force_fields \
  --residue NPOL --stream neopentyl.str > neopentyl.json
```

Pass `--increments /path/to/par_all36_cgenff_increments.prm` if the increment file is stored in another directory, or omit `--stream` to produce only the JSON report. Redirect stdout to a separate `.json` path rather than the input or stream file. Because a failed run can leave an older stream on disk, check the exit status before using the output.

| Exit status | Meaning |
| --- | --- |
| `0` | Assignment or typing completed. For `assign-v5`, the stream was written if requested. |
| `1` | The JSON report contains unresolved items. For `assign-v5`, no stream is written. |
| `2` | Invalid or unsupported input, or a stream export error. When assignment succeeds before export fails, the JSON output includes `stream_error`. |

### Python API

Load the force-field files once to reuse them across multiple ligands:

```python
from pathlib import Path

from autocgenff import V5ForceField, parameterize_v5, write_v5_stream

force_field = V5ForceField.from_directory(Path("force_fields"))
result = parameterize_v5("CC(C)(C)CO", force_field)
if result.is_complete:
    write_v5_stream(result, Path("neopentyl.str"), residue_name="NPOL")
else:
    print(result.issues)
```

For atom typing only, use `from autocgenff.v5 import type_v5`. Both APIs return structured results for programmatic inspection. See [local v5 behavior and limits](docs/local-v5.md) for details.

## Reading the results

- `atoms`: Atom types, exported site charges, and `charge_penalty` for each real atom. Any virtual sites are listed separately in `lone_pairs`.
- `terms`: Assignment status, penalty, and matched source record for each bond, angle, proper torsion, and improper term.
- `charge_diagnostics.terms`: Source, status, and penalty for each charge-increment path.
- `issues`: Unresolved assignments. A `null` value in the report indicates a missing charge or parameter, not zero.

Reported penalties come from the local analogy model and serve as review indicators rather than error estimates or official CGenFF scores. Inspect the assigned parameters and charges before running simulations or free-energy calculations.

In CHARMM, load the compatible base RTF, PRM, and any partner force fields before reading the generated stream, and keep angle and dihedral autogeneration enabled. Coordinates must match the reported atom names and order, as the stream contains neither coordinates nor a PSF. Exact and wildcard parameters, Lennard-Jones terms, and NBFIX entries remain in the base files. See [stream export](docs/local-v5.md#stream-export) for checks and limitations.

## Scope and validation

AutoCGENFF handles small-molecule ligand preparation only. Docking, molecular dynamics, binding free-energy calculations, general topology patches, Drude polarizable models, and PSF writing are outside its scope. Neutral water should use the standard water topology (`toppar_water_ions.str`). Unsupported chemistry, unresolved terms, or ambiguous improper definitions prevent stream export.

The test suite verifies atom typing, charge assignment, bonded analogy, and stream export, with optional read-back checks against `psfgen` and OpenMM. Successful export or a finite single-point energy confirms structural and format validity, not suitability for production simulations.

<details>
<summary>Advanced use: named templates and custom charge models</summary>

The legacy `parameterize` API and CLI can use a named local RTF template selected by the caller:

```sh
autocgenff --prm force_fields/par_all36_cgenff.prm \
  --rtf force_fields/top_all36_cgenff.rtf --smiles CCO --template ETOH
```

For aromatic or charged reference residues, provide independently verified identity with `--template-smiles`. Where equivalent atoms cannot be distinguished by connectivity alone, pass a complete `--atom-map` JSON mapping RTF atom names to explicit atom indices. The legacy workflow also supports caller-supplied SMARTS rules, substitution penalties, and JSON charge increments (`--increments my-increments.json`) with `max_order` and `entries`. For example, `{"types":["A","B"],"values":[0.1]}` transfers 0.1 e from `A` to `B`. Run `autocgenff --help` for full options. Use `assign-v5` for native CGenFF 5.0 increments and checked stream export.

</details>
