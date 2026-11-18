"""Adapt a prepared RDKit molecule to the integer graph consumed by the v5 DSL."""

from dataclasses import dataclass

from rdkit import Chem

from ..errors import AutoCGENFFError
from ._aromaticity import classify_rings
from ._rings import atom_rings, find_rings

Ring = tuple[int, ...]


@dataclass(slots=True)
class Perception:
    elements: list[str]
    charges: list[int]
    neighbors: list[list[int]]
    orders: list[dict[int, int]]
    rings: list[Ring]
    ring_class: dict[Ring, str]
    atom_rings: list[list[Ring]]
    bridgehead: list[bool]

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    def order(self, i: int, j: int) -> int:
        return self.orders[i][j]

    def valence(self, i: int) -> int:
        return sum(self.orders[i].values())


def perceive(molecule: Chem.Mol) -> Perception:
    """Kekulize an explicit-H working copy; reject radicals and ambiguous perception."""
    for atom in molecule.GetAtoms():
        if atom.GetNumRadicalElectrons():
            raise AutoCGENFFError(f"Atom {atom.GetIdx()}: radical chemistry outside local-v5 scope")
    try:
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except (RuntimeError, ValueError) as exc:
        raise AutoCGENFFError(f"Cannot obtain integer Kekule bonds: {exc}") from exc
    elements = [a.GetSymbol() for a in molecule.GetAtoms()]
    charges = [a.GetFormalCharge() for a in molecule.GetAtoms()]
    orders: list[dict[int, int]] = [{} for _ in elements]
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        value = bond.GetBondTypeAsDouble()
        if value not in (1, 2, 3):
            raise AutoCGENFFError(f"Bond {i}-{j}: unsupported bond order {value}")
        orders[i][j] = orders[j][i] = int(value)
    neighbors = [sorted(row) for row in orders]
    rings = find_rings(neighbors)
    classes, _, converged = classify_rings(
        rings, neighbors, elements, lambda i, j: orders[i][j], charges
    )
    if not converged:
        raise AutoCGENFFError("local-v5 aromatic ring classification did not converge")
    member, bridgehead = atom_rings(len(elements), neighbors, rings)
    return Perception(elements, charges, neighbors, orders, rings, classes, member, bridgehead)
