"""Compatibility layer for distributed circuit models.

The functionality has moved to :mod:`dpf2.circuit.distributed`.  This
module re-exports the public API to maintain backwards compatibility
with older import paths.
"""

from .circuit.distributed import TransmissionLineSegment, Switch, assemble_matrices

__all__ = ["TransmissionLineSegment", "Switch", "assemble_matrices"]
