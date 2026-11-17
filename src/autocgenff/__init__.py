"""Independent CGenFF-inspired typing, parameter and charge diagnostics."""

from .analogy import BondGroup, PenaltyModel
from .assignment import AssignmentOptions, AssignmentResult, parameterize
from .charges import ChargeModel
from .errors import AutoCGENFFError
from .molecule import prepare_molecule
from .parameters import read_parameters
from .topology import read_topology
from .typing import TypingRule

__all__ = [
    "AssignmentOptions",
    "AssignmentResult",
    "AutoCGENFFError",
    "BondGroup",
    "ChargeModel",
    "PenaltyModel",
    "TypingRule",
    "parameterize",
    "prepare_molecule",
    "read_parameters",
    "read_topology",
]
