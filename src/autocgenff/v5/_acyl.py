"""Use the neutral carbonyl contributor of localized amide and ester pi pairs."""

from collections import defaultdict

from rdkit import Chem

_SINGLE = Chem.BondType.SINGLE
_DOUBLE = Chem.BondType.DOUBLE
_AROMATIC = Chem.BondType.AROMATIC


def _has_stereo(bond: Chem.Bond) -> bool:
    return bond.GetStereo() != Chem.BondStereo.STEREONONE or bond.GetBondDir() != Chem.BondDir.NONE


def _is_closed_shell_unmarked(atom: Chem.Atom) -> bool:
    return not (
        atom.GetNumRadicalElectrons() or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )


def _is_tail(bond: Chem.Bond, center: Chem.Atom, elements: tuple[int, ...]) -> bool:
    return (
        bond.GetBondType() in (_SINGLE, _AROMATIC)
        and bond.GetOtherAtom(center).GetAtomicNum() in elements
    )


def _is_acyl_donor(bond: Chem.Bond, carbon: Chem.Atom) -> bool:
    if bond.GetBondType() not in (_DOUBLE, _AROMATIC) or _has_stereo(bond):
        return False
    donor = bond.GetOtherAtom(carbon)
    if donor.GetFormalCharge() != 1 or not _is_closed_shell_unmarked(donor):
        return False
    aromatic_pair = bond.GetBondType() == _AROMATIC
    if carbon.GetIsAromatic() != aromatic_pair or donor.GetIsAromatic() != aromatic_pair:
        return False
    tails = [other for other in donor.GetBonds() if other.GetIdx() != bond.GetIdx()]
    # Explicit H makes these degree/order checks valence checks. Require a
    # normal amide (N bound to C/H) or ester (O bound to C), not N–O or an acid.
    match donor.GetAtomicNum(), len(tails):
        case 7, 2:
            allowed = (1, 6)
        case 8, 1:
            allowed = (6,)
        case _:
            return False
    return all(_is_tail(tail, donor, allowed) for tail in tails)


def _acyl_pair(oxide: Chem.Atom) -> tuple[Chem.Bond, Chem.Bond] | None:
    if (oxide.GetAtomicNum(), oxide.GetFormalCharge(), oxide.GetDegree()) != (8, -1, 1):
        return None
    (oxygen_bond,) = oxide.GetBonds()
    if oxygen_bond.GetBondType() != _SINGLE or _has_stereo(oxygen_bond):
        return None
    carbon = oxygen_bond.GetOtherAtom(oxide)
    if (carbon.GetAtomicNum(), carbon.GetFormalCharge(), carbon.GetDegree()) != (6, 0, 3):
        return None
    if not (_is_closed_shell_unmarked(oxide) and _is_closed_shell_unmarked(carbon)):
        return None
    left, right = (bond for bond in carbon.GetBonds() if bond.GetIdx() != oxygen_bond.GetIdx())
    if _is_acyl_donor(left, carbon) and _is_tail(right, carbon, (1, 6)):
        return oxygen_bond, left
    if _is_acyl_donor(right, carbon) and _is_tail(left, carbon, (1, 6)):
        return oxygen_bond, right
    return None


def _neutral_contributor(molecule: Chem.Mol, pairs: list[tuple[int, int, int]]) -> Chem.Mol:
    """Try all acyl edits in a connected component together or leave it unchanged."""
    candidate = Chem.Mol(molecule)
    neutralized: set[int] = set()
    for oxide_index, oxygen_index, donor_index in pairs:
        oxide = candidate.GetAtomWithIdx(oxide_index)
        oxygen_bond = candidate.GetBondWithIdx(oxygen_index)
        carbon = oxygen_bond.GetOtherAtom(oxide)
        donor_bond = candidate.GetBondWithIdx(donor_index)
        donor = donor_bond.GetOtherAtom(carbon)
        if donor.GetIdx() in neutralized:
            # One positive donor cannot neutralize two oxides.
            return molecule
        # Set all local carbonyl constraints before resolving aromatic bonds.
        for atom in (carbon, donor):
            atom.SetIsAromatic(False)
            for bond in atom.GetBonds():
                if bond.GetIsAromatic():
                    if _has_stereo(bond):
                        return molecule
                    bond.SetIsAromatic(False)
                    bond.SetBondType(_SINGLE)
        oxygen_bond.SetBondType(_DOUBLE)
        donor_bond.SetBondType(_SINGLE)
        oxide.SetFormalCharge(0)
        donor.SetFormalCharge(0)
        neutralized.update((oxide_index, donor.GetIdx()))
    if Chem.SanitizeMol(candidate, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return molecule
    for before, after in zip(molecule.GetAtoms(), candidate.GetAtoms(), strict=True):
        expected_charge = 0 if before.GetIdx() in neutralized else before.GetFormalCharge()
        if (
            after.GetFormalCharge() != expected_charge
            or before.GetTotalNumHs() != after.GetTotalNumHs()
            or before.GetNumRadicalElectrons() != after.GetNumRadicalElectrons()
            or before.GetChiralTag() != after.GetChiralTag()
        ):
            return molecule
    for before, after in zip(molecule.GetBonds(), candidate.GetBonds(), strict=True):
        if _has_stereo(before) and (
            before.GetBondType() != after.GetBondType()
            or before.GetStereo() != after.GetStereo()
            or before.GetBondDir() != after.GetBondDir()
            or tuple(before.GetStereoAtoms()) != tuple(after.GetStereoAtoms())
        ):
            return molecule
    return candidate


def normalize_acyl_resonance(molecule: Chem.Mol) -> Chem.Mol:
    """Return the sanitized explicit-H graph with eligible acyl pairs normalized.

    Return the input object if unchanged, otherwise a copy. The paired formal
    charges cancel without moving atoms or H, changing component charge, or
    rewriting bonds bearing stereo. Derived stereo descriptors are refreshed
    on the copy. No mobile anions or cations are inferred.
    """
    candidates: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for component, atoms in enumerate(Chem.GetMolFrags(molecule)):
        for index in atoms:
            pair = _acyl_pair(molecule.GetAtomWithIdx(index))
            if pair is not None:
                oxygen_bond, donor_bond = pair
                candidates[component].append((index, oxygen_bond.GetIdx(), donor_bond.GetIdx()))
    original = molecule
    for pairs in candidates.values():
        # Even edits with distinct donors can compete for the same aromatic
        # electron system. Never commit just the first sanitizable edit.
        molecule = _neutral_contributor(molecule, pairs)
    if molecule is not original:
        # The geometry checks above compare the stereo encoding before and
        # after sanitization. The preserved geometry can acquire new CIP
        # priorities after a bond-order change, so reassign descriptors last.
        Chem.SetDoubleBondNeighborDirections(molecule)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule
