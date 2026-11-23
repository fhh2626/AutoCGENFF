"""Independent CGenFF-inspired typing, parameter and charge diagnostics."""

from ._version import __version__
from .analogy import BondGroup, FallbackModel, PenaltyModel
from .assignment import AssignmentOptions, AssignmentResult, parameterize
from .charges import ChargeModel
from .errors import AutoCGENFFError
from .molecule import prepare_molecule
from .parameters import read_parameters
from .topology import read_topology
from .typing import TypingRule
from .v5_assignment import (
    LonePair,
    V5Assignment,
    V5ForceField,
    bundled_force_field_dir,
    parameterize_v5,
)
from .v5_stream import write_v5_stream

__all__ = [
    "AssignmentOptions",
    "AssignmentResult",
    "AutoCGENFFError",
    "BondGroup",
    "ChargeModel",
    "FallbackModel",
    "LonePair",
    "PenaltyModel",
    "TypingRule",
    "V5Assignment",
    "V5ForceField",
    "__version__",
    "bundled_force_field_dir",
    "parameterize",
    "parameterize_v5",
    "prepare_molecule",
    "read_parameters",
    "read_topology",
    "write_v5_stream",
]
