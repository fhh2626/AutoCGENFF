"""Development-only local edits for damaged B1 SMILES, before atom typing."""

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True, slots=True)
class StructureInput:
    net_charge: int


@dataclass(frozen=True, slots=True)
class StructureResult:
    molecule: Chem.Mol | None
    source: str


def _is_terminal_oxide(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 8 and atom.GetDegree() == 1 and atom.GetFormalCharge() == -1


def _terminal_oxides(atom: Chem.Atom) -> list[Chem.Atom]:
    return [
        o for o in atom.GetNeighbors()
        if _is_terminal_oxide(o)
        and atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), o.GetIdx()).GetBondType()
        == Chem.BondType.SINGLE
    ]


def _finish(mol: Chem.RWMol, target: int) -> Chem.Mol | None:
    out = mol.GetMol()
    if Chem.GetFormalCharge(out) != target or Chem.SanitizeMol(out, catchErrors=True):
        return None
    return None if any(a.GetNumRadicalElectrons() for a in out.GetAtoms()) else out


def _promote_oxide(mol: Chem.RWMol, center: int, oxygen: int) -> None:
    mol.GetBondBetweenAtoms(center, oxygen).SetBondType(Chem.BondType.DOUBLE)
    mol.GetAtomWithIdx(oxygen).SetFormalCharge(0)


def _local_repair(raw: Chem.Mol, target: int, radicals: list[Chem.Atom]) -> StructureResult | None:
    delta = target - Chem.GetFormalCharge(raw)
    if len(radicals) == 1:
        site = radicals[0]
        idx = site.GetIdx()
        if site.GetFormalCharge() or site.GetNumRadicalElectrons() != 1:
            return None
        if delta == -1 and site.GetAtomicNum() == 8 and site.GetDegree() == 1:
            # A bracket [O] at an existing single bond needs one electron, not a relocated pi bond.
            edited = Chem.RWMol(raw)
            edited.GetAtomWithIdx(idx).SetNumRadicalElectrons(0)
            edited.GetAtomWithIdx(idx).SetFormalCharge(-1)
            return StructureResult(_finish(edited, target), "radical-to-oxide")
        if delta == 1 and site.GetAtomicNum() == 6:
            oxygens = _terminal_oxides(site)
            if len(oxygens) == 1:
                # Restore a unique carbonyl without relocating an existing double bond.
                edited = Chem.RWMol(raw)
                edited.GetAtomWithIdx(idx).SetNumRadicalElectrons(0)
                _promote_oxide(edited, idx, oxygens[0].GetIdx())
                return StructureResult(_finish(edited, target), "restore-carbonyl")
    if not radicals and delta == 2:
        choices = []
        for site in raw.GetAtoms():
            idx = site.GetIdx()
            if site.GetAtomicNum() == 7 and site.GetFormalCharge() == -1 and site.GetDegree() == 2:
                oxygens = _terminal_oxides(site)
                has_single_carbon = any(
                    b.GetBondType() == Chem.BondType.SINGLE
                    and b.GetOtherAtom(site).GetAtomicNum() == 6
                    and b.GetOtherAtom(site).GetFormalCharge() == 0
                    for b in site.GetBonds()
                )
                if len(oxygens) == 1 and has_single_carbon:
                    edited = Chem.RWMol(raw)
                    edited.GetAtomWithIdx(idx).SetFormalCharge(0)
                    _promote_oxide(edited, idx, oxygens[0].GetIdx())
                    choices.append(StructureResult(_finish(edited, target), "restore-nitroso"))
            elif site.GetAtomicNum() == 16 and site.GetFormalCharge() == 0 and site.GetDegree() == 4:
                oxygens = _terminal_oxides(site)
                if (
                    len(oxygens) == 3
                    and len({o.GetIsotope() for o in oxygens}) == 1
                    and any(a.GetAtomicNum() == 6 for a in site.GetNeighbors())
                ):
                    # For equivalent O- leaves, input index breaks the resonance tie.
                    # Do not localize charge by index on isotopically distinct leaves.
                    edited = Chem.RWMol(raw)
                    for oxygen in sorted(o.GetIdx() for o in oxygens)[:2]:
                        _promote_oxide(edited, idx, oxygen)
                    choices.append(StructureResult(_finish(edited, target), "restore-sulfonate"))
        if len(choices) == 1:
            return choices[0]
    return None


def prepare_structure(
    raw: Chem.Mol, pinned: Chem.Mol, metadata: StructureInput,
) -> StructureResult:
    """Use only the indexed SMILES graph, residue net charge and pinned B1 fallback.

    Both molecules must have explicit H (as produced by Chem.AddHs) so degree
    checks see the full neighborhood. The caller must exclude RTFs with
    explicit order cards, whose constraints the pinned mapper does not retain.
    Direct bond-order edits are local, but RDKit sanitization may reperceive
    aromaticity elsewhere. A charge-inconsistent pinned result is rejected.
    Neither input is mutated.
    """
    radicals = [a for a in raw.GetAtoms() if a.GetNumRadicalElectrons()]
    if not radicals and Chem.GetFormalCharge(raw) == metadata.net_charge:
        return StructureResult(raw, "unchanged-smiles")
    result = _local_repair(raw, metadata.net_charge, radicals)
    if result is not None and result.molecule is not None:
        return result
    if Chem.GetFormalCharge(pinned) == metadata.net_charge and not any(
        a.GetNumRadicalElectrons() for a in pinned.GetAtoms()
    ):
        return StructureResult(pinned, "pinned-repair")
    return StructureResult(None, "unsupported-repair")
