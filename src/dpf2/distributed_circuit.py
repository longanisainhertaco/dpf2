
"""Compatibility layer for distributed circuit models.

The functionality has moved to :mod:`dpf2.circuit.distributed`.  This
module re-exports the public API to maintain backwards compatibility
with older import paths.
"""

from .circuit.distributed import (
    TransmissionLineSegment,
    TriggeredSwitch,
    assemble_matrices,
    Switch,
)

__all__ = ["TransmissionLineSegment", "TriggeredSwitch", "assemble_matrices", "Switch"]

