
"""Compatibility layer for distributed circuit models.

The functionality has moved to :mod:`dpf2.circuit.distributed`.  This
module re-exports the public API to maintain backwards compatibility
with older import paths.
"""

# Re-export small shims from the new ``dpf2.circuit.distributed`` module so
# existing import paths continue to function.  The real implementations live in
# the submodule but older code expects to import from ``dpf2.distributed_circuit``.
from .circuit.distributed import (
    TransmissionLineSegment,
    TriggeredSwitch,
    CrowbarStage,
    BlumleinSection,
    PlasmaInductance,
    assemble_matrices,
)

# Historically the public API exposed ``Switch`` which is now represented by
# :class:`TriggeredSwitch`.  Provide an explicit alias to maintain backwards
# compatibility with code importing ``Switch`` from this module or the legacy
# path ``dpf2.distributed_circuit``.
Switch = TriggeredSwitch

__all__ = [
    "TransmissionLineSegment",
    "TriggeredSwitch",
    "CrowbarStage",
    "BlumleinSection",
    "PlasmaInductance",
    "assemble_matrices",
    "Switch",
]

