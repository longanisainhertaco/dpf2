"""Models for distributed RLC circuits used in DPF simulations.

This module provides simple dataclasses describing transmission line
segments and switches.  It also exposes helper routines to assemble
R, L and C matrices for a network of segments that can be used with a
state–space integrator.

The implementation is intentionally lightweight – it is not a full
featured circuit simulator but rather a minimal tool to enable unit
testing and examples of distributed networks within the DPF code base.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "TransmissionLineSegment",
    "Switch",
    "assemble_matrices",
]


@dataclass
class TransmissionLineSegment:
    """Simple RLC transmission line segment.

    Parameters are specified per unit length.  ``length`` is measured
    in metres while ``L_per_m``, ``R_per_m`` and ``C_per_m`` are the
    inductance, resistance and capacitance per metre respectively.
    All quantities are converted to total values when building the
    circuit matrices.
    """

    length: float
    L_per_m: float
    R_per_m: float
    C_per_m: float

    def totals(self) -> tuple[float, float, float]:
        """Return the total ``L``, ``R`` and ``C`` for this segment."""

        L = self.L_per_m * self.length
        R = self.R_per_m * self.length
        C = self.C_per_m * self.length
        return L, R, C


@dataclass
class Switch:
    """Idealised switch model.

    The switch is represented only by an on and off resistance.  The
    state of the switch is toggled by setting ``closed`` to ``True`` or
    ``False``.
    """

    closed: bool = True
    R_on: float = 1e-3
    R_off: float = 1e6

    def resistance(self) -> float:
        """Return the instantaneous resistance of the switch."""

        return self.R_on if self.closed else self.R_off


def assemble_matrices(
    segments: Sequence[TransmissionLineSegment],
    switches: Sequence[Switch] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble diagonal ``R``, ``L`` and ``C`` matrices for a network.

    Each segment contributes its total ``R``, ``L`` and ``C`` to the
    diagonal of the returned matrices.  If ``switches`` are provided
    their resistances are added in series with the corresponding
    segments.

    The matrices are primarily intended for use with the simple
    state–space integrator implemented in :mod:`dpf2.circuit_solver`.
    They are not intended to cover all possible circuit topologies but
    are sufficient for modelling a linear chain of segments and
    switches.
    """

    n = len(segments)
    R = np.zeros((n, n), dtype=float)
    L = np.zeros((n, n), dtype=float)
    C = np.zeros((n, n), dtype=float)

    for i, seg in enumerate(segments):
        L_tot, R_tot, C_tot = seg.totals()
        L[i, i] = L_tot
        R[i, i] = R_tot
        C[i, i] = C_tot

    if switches is not None:
        for i, sw in enumerate(switches):
            if i < n:
                R[i, i] += sw.resistance()

    return R, L, C
