"""Offline local-v5 atom typing, independent of force-field and template assignment."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from rdkit import Chem

from ..molecule import prepare_molecule, reject_ligand_water
from ._engine import type_perception
from ._perception import perceive
from ._rules import RuleSet, parse


@dataclass(frozen=True, slots=True)
class V5Atom:
    """One explicit atom; trace entries identify source categories, lines and jumps."""

    index: int
    element: str
    formal_charge: int
    rule_charge: int
    improper_center: bool
    atom_type: str | None
    trace: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class V5Result:
    """Diagnostic assignments only; absent types are not usable as parameters."""

    atoms: tuple[V5Atom, ...]
    profile: str = "local-v5"

    @property
    def types(self) -> tuple[str | None, ...]:
        return tuple(atom.atom_type for atom in self.atoms)

    @property
    def is_complete(self) -> bool:
        return all(atom.atom_type is not None for atom in self.atoms)


@lru_cache(maxsize=1)
def _profile() -> RuleSet:
    # Only the immutable installed profile is supported, not user-supplied programs.
    name = "cgenff_v5.rules"
    return parse(files(__package__).joinpath(name).read_text(encoding="utf-8"), name)


def v5_vocabulary() -> frozenset[str]:
    """Static typ targets, expanding both digits of altnum leaves (not a coverage claim)."""
    labels = set()
    for rules in _profile().categories.values():
        for rule in rules:
            if rule.action == "typ":
                if rule.altnum:
                    labels.update(rule.target.replace("?", digit) for digit in ("1", "2"))
                else:
                    labels.add(rule.target)
    return frozenset(labels)


def type_v5(source: str | Path | Chem.Mol) -> V5Result:
    """Type a SMILES, single SDF/MOL2 path or RDKit molecule without mutating it.

    Indices are zero-based; existing explicit atoms retain their order, added H are
    appended. Check is_complete: no matching branch returns None and a reason on
    that atom. Invalid structures, radicals, nonconvergence, circular dependencies
    and conflicting alternating numbers raise AutoCGENFFError. Formal charges are
    from the sanitized input, not partial charges or DSL charge actions.
    """
    return _type_prepared(prepare_molecule(source))


def _type_prepared(molecule: Chem.Mol) -> V5Result:
    reject_ligand_water(molecule)
    perception = perceive(molecule)
    typed = type_perception(_profile(), perception)
    atoms = tuple(
        V5Atom(
            i,
            perception.elements[i],
            perception.charges[i],
            row.rule_charge,
            row.improper_center,
            None if row.errors or row.type == "UNK" else row.type,
            tuple(str(rule) for rule in row.trace),
            tuple(row.errors),
            tuple(row.warnings),
        )
        for i, row in enumerate(typed)
    )
    return V5Result(atoms)
