"""Interpreter for the typing rule language (P-I; SPEC 5.5) on a Perception.

Typing of each atom starts at `cat main`; in each category the first rule whose conditions all hold fires. `ne`
matches its groups greedily and in order against the not-yet-used neighbours, lowest atom index first, without
backtracking. Inside a group evaluated on a neighbour, `bo`/`inring` refer to the bond just traversed, and nested
`ne` does not go back over that bond (interpretation: otherwise `ne ( ne ( el C ) )` would trivially match the atom
being typed; `self` then detects ring closures). A ring matched by a ring keyword is consumed for the rest of that
group, so `arom 6 arom 6` needs two aromatic 6-rings. Ring keywords look at the atom's ring list (smallest ring, or
the three smallest for bridgeheads, SPEC 5.2); `inring` and `rings` as noted in rules.py.
"""

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from itertools import permutations

from ..errors import AutoCGENFFError
from ._perception import Perception, Ring
from ._rules import Rule, RuleSet

HALOGENS = frozenset(("F", "Cl", "Br", "I"))
MAX_SUB_DEPTH = 64


class TypingError(AutoCGENFFError):
    pass


@dataclass
class AtomTyping:
    type: str = "UNK"
    altnum: bool = False
    rule_charge: int = 0
    improper_center: bool = False
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    trace: list[Rule] = field(default_factory=list)  # rules that fired, main first


class _Evaluator:
    def __init__(self, rules, perception):
        p = perception
        self.rules, self.p = rules, p
        self.el = p.elements
        self.nbrs = p.neighbors
        self.valence = [p.valence(i) for i in range(p.n_atoms)]
        self.n_h = [sum(self.el[j] == "H" for j in p.neighbors[i]) for i in range(p.n_atoms)]
        self.bond_rings: dict[frozenset[int], list[Ring]] = {}
        for r in p.rings:
            for k in range(len(r)):
                self.bond_rings.setdefault(frozenset((r[k], r[(k + 1) % len(r)])), []).append(r)
        # fusion atom: bridgehead (SPEC 5.2) two of whose rings share at least one more atom (spiro atoms are not)
        # and that has >= 3 neighbours in the two rings together
        self.fusion = [
            p.bridgehead[i]
            and any(
                len(set(r1) & set(r2)) >= 2
                and sum(j in set(r1) | set(r2) for j in p.neighbors[i]) >= 3
                for x, r1 in enumerate(p.atom_rings[i])
                for r2 in p.atom_rings[i][x + 1 :]
            )
            for i in range(p.n_atoms)
        ]
        self.typed = [None] * p.n_atoms
        self.active = set()
        self.root = None

    def _ring_ok(self, r, cls, size):
        return (cls is None or self.p.ring_class[r] == cls) and (size is None or len(r) == size)

    def match(self, group, a, came, consumed):
        return all(self.cond(c, a, came, consumed) for c in group)

    def _match_neighbors(self, groups, a, came, neighbors):
        used = set()
        for group in groups:
            for b in neighbors:
                if b != came and b not in used and self.match(group, b, a, set()):
                    used.add(b)
                    break
            else:
                return False
        return True

    def cond(self, c, a, came, consumed):
        k = c[0]
        if k == "el":
            return self.el[a] == c[1]
        if k == "elha":
            return self.el[a] in HALOGENS
        if k == "elos":
            return self.el[a] in ("O", "S")
        if k == "nb":
            return self.valence[a] == c[1]
        if k == "nh":
            return self.n_h[a] == c[1]
        if k == "deg":
            return len(self.nbrs[a]) == c[1]
        if k == "fc":
            return self.p.charges[a] == c[1]
        if k == "rings":
            return len(self.p.atom_rings[a]) == c[1]
        if k == "ringkw":
            for r in self.p.atom_rings[a]:
                if r not in consumed and self._ring_ok(r, c[1], c[2]):
                    consumed.add(r)
                    return True
            return False
        if k == "self":
            return a == self.root
        if k == "fusion":
            return self.fusion[a]
        if k == "guanidinium_ring_imine":
            witness = _guanidinium_ring_imine(self.p, a, self.bond_rings)
            if witness is None:
                return False
            center, source = witness
            return self.iminium_source(center, self.type_atom(center)) == source
        if k == "ne":
            return self._match_neighbors(c[1], a, came, self.nbrs[a])
        if k == "rne":
            for ring in self.p.atom_rings[a]:
                if not self._ring_ok(ring, c[1], c[2]):
                    continue
                i = ring.index(a)
                perimeter = tuple(b for b in (ring[i - 1], ring[(i + 1) % len(ring)]) if b != came)
                groups = c[3]
                if any(
                    all(self.match(group, b, a, set()) for group, b in zip(groups, order, strict=True))
                    for order in permutations(perimeter, len(groups))
                ):
                    return True
            return False
        if k == "bo":
            return came is not None and self.p.order(came, a) == c[1]
        if k == "inring":
            if came is None:
                return False
            return any(
                self._ring_ok(r, c[1], c[2]) for r in self.bond_rings.get(frozenset((came, a)), ())
            )
        if k == "not":
            return not self.match(c[1], a, came, set(consumed))
        if k == "or":
            for g in c[1]:
                trial = set(consumed)
                if self.match(g, a, came, trial):
                    consumed |= trial
                    return True
            return False
        if k == "type":
            return fnmatchcase(self.type_atom(a).type, c[1])
        if k == "rmember":
            return any(
                self.match(c[3], m, None, set())
                for r in self.p.atom_rings[a]
                if self._ring_ok(r, c[1], c[2])
                for m in r
            )
        raise TypingError(f"unknown condition {k}")

    def type_atom(self, i):
        if self.typed[i] is not None:
            return self.typed[i]
        if i in self.active:
            raise TypingError(f"atom {i}: circular 'type' dependency")
        self.active.add(i)
        saved_root, self.root = self.root, i
        try:
            result = self._run(i)
        finally:
            self.root = saved_root
            self.active.discard(i)
        self.typed[i] = result
        return result

    def _run(self, i):
        out = AtomTyping()
        cat = "main"
        for _ in range(MAX_SUB_DEPTH):
            for rule in self.rules.categories[cat]:
                if self.match(rule.conditions, i, None, set()):
                    break
            else:
                out.errors.append(f"no rule matches in category {cat}")
                out.type = "UNK"
                return out
            out.trace.append(rule)
            if rule.charge is not None:
                out.rule_charge = rule.charge
            out.improper_center |= rule.impr
            out.altnum |= rule.altnum
            out.warnings += rule.warn
            out.errors += rule.err
            if rule.action == "typ":
                out.type = rule.target
                return out
            cat = rule.target
        raise TypingError(f"atom {i}: more than {MAX_SUB_DEPTH} nested 'sub' jumps")

    def iminium_source(self, carbon: int, typed: AtomTyping) -> int | None:
        """Attribute the final charge action only if complete neighbor matches agree on its N."""
        rule = next((rule for rule in reversed(typed.trace) if rule.charge is not None), None)
        if rule is None or not rule.iminium_charge or typed.errors:
            return None
        source_groups = []
        for condition in rule.conditions:
            if condition[0] == "ne":
                groups = condition[1]
                for index, group in enumerate(groups):
                    if ("el", "N") in group and ("bo", 2) in group:
                        source_groups.append((groups, index))
        if len(source_groups) != 1:
            return None
        groups, source_index = source_groups[0]
        # Replay the source's ne with distinct neighbors, without relying on the greedy tie-break.
        candidates = set()
        saved_root, self.root = self.root, carbon
        try:
            for order in permutations(self.nbrs[carbon], len(groups)):
                source = order[source_index]
                if source not in candidates and all(
                    self.match(group, neighbor, carbon, set())
                    for group, neighbor in zip(groups, order, strict=True)
                ):
                    candidates.add(source)
                    if len(candidates) > 1:
                        return None
        finally:
            self.root = saved_root
        return next(iter(candidates), None)


def _resolve_altnum(types, atoms, perception):
    """Replace '?' by 1/2: atoms joined by a double bond share the digit, by a single bond they alternate (SPEC 5.5).
    Each conjugated chain is numbered from its lowest-index atom, which gets 1."""
    todo = {i for i, a in enumerate(atoms) if a.altnum and "?" in a.type}
    digit = {}
    for start in sorted(todo):
        if start in digit:
            continue
        digit[start] = 1
        stack = [start]
        while stack:
            i = stack.pop()
            for j in perception.neighbors[i]:
                if j not in todo:
                    continue
                want = digit[i] if perception.order(i, j) >= 2 else 3 - digit[i]
                if j not in digit:
                    digit[j] = want
                    stack.append(j)
                elif digit[j] != want:
                    raise TypingError(f"altnum: inconsistent numbering at atoms {i}-{j}")
    out = list(types)
    for i, d in digit.items():
        out[i] = out[i].replace("?", str(d), 1)
    return out


def _is_guanidinium_center(p: Perception, carbon: int, nitrogen: int) -> bool:
    """Use C=N+ and two C–N bonds at neutral C as a structural proxy only."""
    bonds = p.orders[carbon]
    return (
        p.charges[carbon] == 0
        and len(bonds) == 3
        and all(
            p.elements[j] == "N" and order == (2 if j == nitrogen else 1)
            for j, order in bonds.items()
        )
    )


def _is_amine_donor(p: Perception, donor: int, center: int) -> bool:
    """Require H or saturated carbon outside the guanidinium group.

    This conservative witness avoids deciding whether a lone pair also
    conjugated with an alkene, aryl group, or carbonyl can represent the cation.
    The substituents stay the same when the donor bears the C=N+ bond.
    """
    return all(
        p.charges[j] == 0 and (
            p.elements[j] == "H" or (
                p.elements[j] == "C" and len(p.orders[j]) == 4 and p.valence(j) == 4
            )
        )
        for j in p.neighbors[donor] if j != center
    )


def _guanidinium_ring_imine(
    p: Perception, n: int, bond_rings: dict[frozenset[int], list[Ring]]
) -> tuple[int, int] | None:
    """Find the guanidinium carbon and its drawn N+ source, without editing the graph."""
    bonds = p.orders[n]
    charge = p.charges[n]
    if p.elements[n] != "N" or charge not in (0, 1) or len(bonds) != 2:
        return None
    if not all(p.elements[j] == "C" for j in bonds) or sorted(bonds.values()) != [1 + charge, 2]:
        return None

    for center, link in bonds.items():
        if link != 1 + charge:
            continue
        imine = next(j for j in bonds if j != center)
        # The retained C=N must be trigonal and not part of a competing cationic group.
        imine_bonds = p.orders[imine]
        if (
            p.charges[imine] != 0
            or sorted(imine_bonds.values()) != [1, 1, 2]
            or all(p.elements[j] == "N" for j in imine_bonds)
            or any(
                order == 1 and p.elements[j] == "N" and p.charges[j] == 0
                and len(p.orders[j]) == 3 and p.valence(j) == 3
                for j, order in imine_bonds.items()
            )
        ):
            continue
        source = next((j for j, order in p.orders[center].items() if order == 2), None)
        if source is None or not _is_guanidinium_center(p, center, source) or p.charges[source] != 1:
            continue
        # Use the actual retained C=N edge. With degree two at N, both its
        # bonds lie on that same perimeter. Do not override another ring charge.
        if not any(
            len(r) == 5 and all(p.charges[j] == 0 for j in r if j != source)
            for r in bond_rings.get(frozenset((n, imine)), ())
        ):
            continue
        flanks = [j for j in p.neighbors[center] if j != n]
        # The remaining Ns are ordinary three-coordinate amino nitrogens:
        # neutral single-bonded flanks and, in the neutral drawing, one =N+.
        if not all(
            len(p.orders[j]) == 3 and p.charges[j] == int(j == source)
            and p.valence(j) == 3 + p.charges[j]
            for j in flanks
        ):
            continue
        if any(_is_amine_donor(p, j, center) for j in flanks):
            return center, source
    return None


def _resolve_shared_iminium_seeds(ev: _Evaluator, atoms: list[AtomTyping]) -> None:
    """Drop a duplicate seed at the non-guanidinium end of C=N+=C.

    Fig. 5 motivates placing a guanidinium seed on carbon. Here the structural
    proxy only breaks a tie in seed ownership; it does not establish resonance
    delocalization. Leave unproven or ambiguous claims to charge validation.
    """
    p = ev.p
    for n, bonds in enumerate(p.orders):
        if p.elements[n] != "N" or p.charges[n] != 1 or len(bonds) != 2:
            continue
        if atoms[n].rule_charge or atoms[n].errors:
            continue
        if not all(order == 2 and p.elements[c] == "C" for c, order in bonds.items()):
            continue
        left, right = bonds
        left_center = _is_guanidinium_center(p, left, n)
        right_center = _is_guanidinium_center(p, right, n)
        if left_center == right_center:
            continue
        if all(atoms[c].rule_charge == 1 and ev.iminium_source(c, atoms[c]) == n for c in bonds):
            atoms[right if left_center else left].rule_charge = 0


def type_perception(rules: RuleSet, perception: Perception) -> list[AtomTyping]:
    """Evaluate all branches before resolving shared seeds and alternating digits."""
    ev = _Evaluator(rules, perception)
    atoms = [ev.type_atom(i) for i in range(perception.n_atoms)]
    _resolve_shared_iminium_seeds(ev, atoms)
    types = _resolve_altnum([a.type for a in atoms], atoms, perception)
    for atom, label in zip(atoms, types, strict=True):
        atom.type = label
    return atoms
