"""Models for distributed RLC circuits used in DPF simulations.


This module defines lightweight data structures describing
transmission line segments and switches.  The intent is to provide a
minimal representation that can be translated from configuration data
and consumed by simple network assembly routines.  The models here are
not intended to be a full-featured circuit simulator.
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
    """Simple RLC transmission line segment with optional parasitics.

    Parameters are specified per unit length.  ``length`` is measured in
    metres while ``L_per_m``, ``R_per_m`` and ``C_per_m`` are the
    inductance, resistance and capacitance per metre respectively.  In
    addition, fixed parasitic ``L``, ``R`` and ``C`` can be specified for
    the whole segment.  ``from_node`` and ``to_node`` identify the nodes
    connected by the segment.
    """

    from_node: int
    to_node: int

    length: float
    L_per_m: float
    R_per_m: float
    C_per_m: float

    L_parasitic: float = 0.0
    R_parasitic: float = 0.0
    C_parasitic: float = 0.0
    L_profile: Sequence[tuple[float, float]] | None = None
    R_profile: Sequence[tuple[float, float]] | None = None
    C_profile: Sequence[tuple[float, float]] | None = None


    def totals(self) -> tuple[float, float, float]:
        """Return the total ``L``, ``R`` and ``C`` for this segment."""


        L = self.L_per_m * self.length + self.L_parasitic
        R = self.R_per_m * self.length + self.R_parasitic
        C = self.C_per_m * self.length + self.C_parasitic

        return L, R, C


@dataclass

class Switch:
    """Idealised switch model with optional trigger times and parasitics.

    ``from_node`` and ``to_node`` identify the nodes bridged by the
    switch.  ``trigger_times`` is an optional sequence of times in
    seconds when the switch state should toggle.  ``L_parasitic`` and
    ``C_parasitic`` provide fixed inductive and capacitive contributions
    irrespective of the switch state while ``R_parasitic`` is a series
    resistance always present in addition to the state-dependent value
    returned by :meth:`resistance`.
    """

    from_node: int
    to_node: int
    closed: bool = True
    R_on: float = 1e-3
    R_off: float = 1e6
    trigger_times: Sequence[float] | None = None
    L_parasitic: float = 0.0
    R_parasitic: float = 0.0
    C_parasitic: float = 0.0

    def resistance(self, t: float | None = None) -> float:
        """Return the instantaneous resistance of the switch."""

        base = self.R_on if self.closed else self.R_off
        return base + self.R_parasitic


def assemble_matrices(
    segments: Sequence[TransmissionLineSegment],

    switches: Sequence[Switch] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble diagonal ``R``, ``L`` and ``C`` matrices for a network.

    Each segment contributes its total ``R``, ``L`` and ``C`` to the
    diagonal of the returned matrices.  If ``switches`` are provided
    their resistances are added in series with the corresponding
    segments.  Node connections are ignored in this simple assembly
    routine which assumes a linear chain of segments.
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
                L[i, i] += sw.L_parasitic
                C[i, i] += sw.C_parasitic


    return R, L, C
