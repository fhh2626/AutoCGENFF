"""Staged regularized sparse least-squares fitting from caller-owned references."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
from rdkit import Chem
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .charges import PATH_ORDER, ChargeModel, _check_charge_totals, formal_baseline
from .errors import AutoCGENFFError
from .molecule import inventory
from .parameters import Types


@dataclass(frozen=True, slots=True)
class TypedReference:
    """Prepared explicit-H molecule with vetted types, target charges and baseline."""

    molecule: Chem.Mol
    types: Types
    charges: tuple[float, ...]
    baseline: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class FitStage:
    """Diagnostics after fitting one path order to the previous stage's residual."""

    order: int
    columns: int
    rms_error: float


@dataclass(frozen=True, slots=True)
class FitResult:
    """The increment table and RMS errors on training targets (not validation)."""

    model: ChargeModel
    stages: tuple[FitStage, ...]


def fit_increments(
    references: Sequence[TypedReference], *, max_order: int = 4, ridge: float = 0.001
) -> FitResult:
    """Fit independent bond, angle and torsion stages with earlier stages frozen.

    Fully palindromic paths are fixed at zero in this fit. `ridge` is the
    coefficient on the sum of squared fitted increments, not its square root.
    Invalid inputs and failure to converge raise AutoCGENFFError.
    """
    if type(max_order) is not int or max_order not in (2, 3, 4):
        raise AutoCGENFFError("Fit max_order must be an integer (2, 3 or 4)")
    if not isfinite(ridge) or ridge < 0:
        raise AutoCGENFFError("Fit ridge must be finite and nonnegative")
    if not references:
        raise AutoCGENFFError("At least one typed reference is required")
    paths: dict[int, list[tuple[Types, tuple[int, ...]]]] = {2: [], 3: [], 4: []}
    differences = []
    for reference in references:
        molecule = reference.molecule
        count = molecule.GetNumAtoms()
        if (
            not count
            or len(reference.types) != count
            or len(reference.charges) != count
            or not all(isfinite(charge) for charge in reference.charges)
        ):
            raise AutoCGENFFError("Reference requires one type and a finite target charge per atom")
        if any(atom.GetNumImplicitHs() or atom.GetNumExplicitHs() for atom in molecule.GetAtoms()):
            raise AutoCGENFFError(
                "Fitting references must have all hydrogen atoms explicitly present"
            )
        start = formal_baseline(molecule, reference.baseline)
        _check_charge_totals(
            molecule,
            reference.charges,
            start,
            "Reference target charge total disagrees with baseline",
        )
        offset = len(differences)
        differences.extend(
            charge - value for charge, value in zip(reference.charges, start, strict=True)
        )
        for term in inventory(molecule):
            types = tuple(reference.types[index] for index in term.atoms)
            indices = tuple(index + offset for index in term.atoms)
            if types > types[::-1]:
                types, indices = types[::-1], indices[::-1]
            paths[PATH_ORDER[term.kind]].append((types, indices))
    residual = np.array(differences, dtype=float)
    increments = {}
    stages = []
    for order in range(2, max_order + 1):
        keys = sorted({types for types, _ in paths[order]})
        active = [types for types in keys if types != types[::-1]]
        if active:
            design = _design_matrix(paths[order], active, len(residual), order)
            values, stop, *_ = lsqr(design, residual, damp=sqrt(ridge), atol=1e-10, btol=1e-10)
            if stop not in (0, 1, 2, 4, 5):
                raise AutoCGENFFError(
                    f"Least-squares stage {order} did not converge (LSQR status {stop})"
                )
            residual -= design @ values
            for index, key in enumerate(active):
                first = index * (order - 1)
                increments[key] = tuple(float(value) for value in values[first : first + order - 1])
        for key in keys:
            increments.setdefault(key, (0.0,) * (order - 1))
        stages.append(
            FitStage(order, len(active) * (order - 1), float(np.sqrt(np.mean(residual**2))))
        )
    return FitResult(ChargeModel(increments, max_order=max_order), tuple(stages))


def _design_matrix(
    paths: list[tuple[Types, tuple[int, ...]]], active: list[Types], rows: int, order: int
) -> coo_matrix:
    columns = {key: index * (order - 1) for index, key in enumerate(active)}
    row_indices = []
    col_indices = []
    data = []
    for key, indices in paths:
        if key not in columns:
            continue
        for edge in range(order - 1):
            row_indices.extend((indices[edge], indices[edge + 1]))
            col_indices.extend((columns[key] + edge,) * 2)
            data.extend((-1.0, 1.0))
    return coo_matrix(
        (data, (row_indices, col_indices)), shape=(rows, len(active) * (order - 1))
    ).tocsr()
