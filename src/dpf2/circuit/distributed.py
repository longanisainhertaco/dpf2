"""Models for distributed RLC circuits used in DPF simulations.

This module provides dataclasses describing transmission line segments,
triggered switches and parasitic elements. It also exposes helper routines
that assemble R, L and C matrices for a network that can be used with a
state–space integrator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

__all__ = [
    "TransmissionLineSegment",
    "TriggeredSwitch",
    "ShuntCapacitance",
    "StrayInductance",
    "assemble_matrices",
]


class Parasitic(Protocol):
    """Protocol for parasitic elements attachable to a segment."""

    def totals(self) -> tuple[float, float, float]:
        """Return contributions to ``L``, ``R`` and ``C``."""


@dataclass
class ShuntCapacitance:
    """Simple shunt capacitance parasitic."""

    C: float

    def totals(self) -> tuple[float, float, float]:
        return 0.0, 0.0, self.C


@dataclass
class StrayInductance:
    """Simple stray inductance parasitic."""

    L: float

    def totals(self) -> tuple[float, float, float]:
        return self.L, 0.0, 0.0


@dataclass
class TransmissionLineSegment:
    """RLC transmission line segment with optional parasitics.

    Parameters are specified per unit length. ``length`` is measured in
    metres while ``L_per_m``, ``R_per_m`` and ``C_per_m`` are the inductance,
    resistance and capacitance per metre respectively.  Parasitic elements
    can be attached and are included when computing the total values.
    """

    length: float
    L_per_m: float
    R_per_m: float
    C_per_m: float
    parasitics: Sequence[Parasitic] = field(default_factory=tuple)

    def totals(self) -> tuple[float, float, float]:
        """Return the total ``L``, ``R`` and ``C`` for this segment."""

        L = self.L_per_m * self.length
        R = self.R_per_m * self.length
        C = self.C_per_m * self.length
        for p in self.parasitics:
            pL, pR, pC = p.totals()
            L += pL
            R += pR
            C += pC
        return L, R, C


@dataclass
class TriggeredSwitch:
    """Switch that closes at a specified trigger time."""

    trigger_time: float
    R_on: float = 1e-3
    R_off: float = 1e6

    def resistance(self, time: float) -> float:
        """Return the instantaneous resistance at ``time``."""

        return self.R_on if time >= self.trigger_time else self.R_off


def assemble_matrices(
    segments: Sequence[TransmissionLineSegment],
    switches: Sequence[TriggeredSwitch] | None = None,
    time: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble diagonal ``R``, ``L`` and ``C`` matrices for a network.

    Each segment contributes its total ``R``, ``L`` and ``C`` (including
    parasitic elements) to the diagonal of the returned matrices.  If
    ``switches`` are provided their resistances at ``time`` are added in
    series with the corresponding segments.
    """

    n = len(segments)
    R = np.zeros((n, n))
    L = np.zeros((n, n))
    C = np.zeros((n, n))

    for i, seg in enumerate(segments):
        L_tot, R_tot, C_tot = seg.totals()
        L[i, i] = L_tot
        R[i, i] = R_tot
        C[i, i] = C_tot

    if switches is not None:
        for i, sw in enumerate(switches):
            if i < n:
                R[i, i] += sw.resistance(time)

    return R, L, C
