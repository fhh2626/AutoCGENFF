"""Prioritized SMARTS atom typing for a deliberately small organic domain."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from rdkit import Chem

from .errors import AutoCGENFFError
from .molecule import reject_ligand_water


@dataclass(frozen=True, slots=True)
class TypingRule:
    """Assign `atom_type` to the `target` query atom, highest priority first.

    The first rule wins ties. `target` is a zero-based SMARTS query position.
    Invalid queries raise AutoCGENFFError during construction.
    """

    name: str
    smarts: str
    atom_type: str
    priority: int = 0
    target: int = 0
    _query: Chem.Mol = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.priority) is not int or type(self.target) is not int:
            raise AutoCGENFFError("Typing rule priority and target must be integers")
        if not all(
            isinstance(v, str) and v.strip() for v in (self.name, self.smarts, self.atom_type)
        ):
            raise AutoCGENFFError("Typing rule name, SMARTS and type must be nonempty strings")
        query = Chem.MolFromSmarts(self.smarts)
        if query is None or not 0 <= self.target < query.GetNumAtoms():
            raise AutoCGENFFError(f"Invalid SMARTS or target in typing rule {self.name!r}")
        if (
            self.atom_type.upper() == "X"
            or any(char.isspace() for char in self.atom_type)
        ):
            raise AutoCGENFFError(f"Invalid atom type in typing rule {self.name!r}")
        object.__setattr__(self, "_query", query)


@dataclass(frozen=True, slots=True)
class TypedAtom:
    """Assigned CHARMM type and the rule or named template that supplied it."""

    atom_type: str
    source: str


# Local rules are independent of the official rule tree. Recursive SMARTS
# guard the functional environment so unreviewed chemistry stays untyped.
_ALKYL = "[C+0;X4;!R;!$(*~[!#1;!#6;!#8;!#7]);!$(*-[O;!$([O+0;X2;H1]-[#6])]);!$(*-[N;!$([N+;X4;H3]-[#6])]);!$(*-[#6;!X4])]"
_ALCOHOL = f"[O+0;X2;H1;$(*-{_ALKYL})]"
_AMMONIUM = f"[N+;X4;H3;$(*-{_ALKYL})]"
_BENZENE = "[c+0;r6;R1;H1]1[c+0;r6;R1;H1][c+0;r6;R1;H1][c+0;r6;R1;H1][c+0;r6;R1;H1][c+0;r6;R1;H1]1"
_OLEFINIC_CARBON = "C+0;X3;H1,H2;!R;!$(*~[!#1;!#6]);!$(*-[#6;!X4])"
_ALKENE = f"[{_OLEFINIC_CARBON};$(*=[{_OLEFINIC_CARBON}])]"
_ALLYLIC = f"[C+0;X4;!R;!$(*~[!#1;!#6]);$(*-{_ALKENE});!$(*-[#6;!X4;!$({_ALKENE})])]"
_NEUTRAL_ALKYL = f"[$({_ALKYL}),$({_ALLYLIC})]"


def _with_hydrogens(name: str, smarts: str, atom_type: str, h_type: str) -> tuple[TypingRule, ...]:
    return (TypingRule(name, smarts, atom_type), TypingRule(f"{name}-H", f"[#1]-{smarts}", h_type))


def _builtin_rules() -> tuple[TypingRule, ...]:
    rules = [TypingRule("alkyl-C0H", f"[{_NEUTRAL_ALKYL[1:-1]};H0;!$(*-[O,N])]", "CG301")]
    for count, carbon in enumerate(("CG311", "CG321", "CG331"), 1):
        smarts = f"[{_NEUTRAL_ALKYL[1:-1]};H{count};!$(*-[N+])]"
        rules.extend(_with_hydrogens(f"alkyl-C{count}H", smarts, carbon, f"HGA{count}"))
    for count, carbon in ((1, "CG314"), (2, "CG324"), (3, "CG334")):
        smarts = f"[{_ALKYL[1:-1]};H{count};$(*-[N+;X4;H3]);!$(*-O)]"
        rules.extend(_with_hydrogens(f"ammonium-adjacent-C{count}H", smarts, carbon, f"HGA{count}"))
    rules.extend(_with_hydrogens("alcohol-O", _ALCOHOL, "OG311", "HGP1"))
    rules.extend(_with_hydrogens("primary-alkylammonium-N", _AMMONIUM, "NG3P3", "HGP2"))
    rules.extend(_with_hydrogens("benzene-C", _BENZENE, "CG2R61", "HGR61"))
    for count, carbon, hydrogen in ((1, "CG2D1", "HGA4"), (2, "CG2D2", "HGA5")):
        smarts = f"[{_ALKENE[1:-1]};H{count}]"
        rules.extend(_with_hydrogens(f"alkene-C{count}H", smarts, carbon, hydrogen))
    return tuple(rules)


BUILTIN_RULES = _builtin_rules()


def type_atoms(
    molecule: Chem.Mol, rules: Sequence[TypingRule] = (), *, include_builtin: bool = True
) -> tuple[TypedAtom, ...]:
    """Apply prioritized rules to a prepared molecule; raises for any untyped atom."""
    reject_ligand_water(molecule)
    candidates = (*rules, *BUILTIN_RULES) if include_builtin else tuple(rules)
    assignments: list[TypedAtom | None] = [None] * molecule.GetNumAtoms()
    for rule in sorted(candidates, key=lambda item: -item.priority):
        options = Chem.SubstructMatchParameters()
        options.useChirality = True
        options.uniquify = False  # Same atom set can have different target positions.
        options.maxMatches = 100_000
        matches = molecule.GetSubstructMatches(rule._query, options)
        if len(matches) == options.maxMatches:
            raise AutoCGENFFError(f"SMARTS match limit reached for rule {rule.name!r}")
        for match in matches:
            index = match[rule.target]
            if assignments[index] is None:
                assignments[index] = TypedAtom(rule.atom_type.upper(), rule.name)
    unknown = [
        f"{index}:{molecule.GetAtomWithIdx(index).GetSymbol()}"
        for index, assignment in enumerate(assignments)
        if assignment is None
    ]
    if unknown:
        raise AutoCGENFFError(f"No atom typing rule for indices {', '.join(unknown)}")
    return tuple(assignment for assignment in assignments if assignment is not None)
