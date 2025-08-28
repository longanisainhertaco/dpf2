"""Backward compatibility wrapper for distributed circuit models.

The implementations have moved to :mod:`dpf2.circuit.distributed`.
"""
from .circuit.distributed import (
    TransmissionLineSegment,
    TriggeredSwitch,
    ShuntCapacitance,
    StrayInductance,
    assemble_matrices,
)

# ``Switch`` was renamed to ``TriggeredSwitch``; provide an alias for
# existing callers.
Switch = TriggeredSwitch

__all__ = [
    "TransmissionLineSegment",
    "TriggeredSwitch",
    "ShuntCapacitance",
    "StrayInductance",
    "assemble_matrices",
    "Switch",
]
