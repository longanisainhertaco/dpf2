from __future__ import annotations

"""Light‑weight representations of distributed circuit components.

The real project contains a sophisticated distributed circuit solver.  For the
purposes of the exercises in this kata we implement only the minimal features
required by the tests.  The goal is to provide simple data classes describing
transmission line segments and switches together with a helper to assemble
system matrices used by the time integrator in :mod:`dpf2.rlc_solver`.

Each :class:`TransmissionLineSegment` stores per–unit length parameters and
optional parasitic components that apply to the whole segment.  Time dependant
profiles can be supplied for the R, L and C values in the form of ``[(t, val)]``
pairs.  The :class:`TriggeredSwitch` models an ideal resistive switch which can
change state at user supplied trigger times and may also have fixed parasitic
components attached.

The :func:`assemble_matrices` function converts a list of segments and switches
into diagonal R, L and C matrices.  The matrices are intentionally extremely
simple – topology is ignored and all elements are assumed to be in series –
which is perfectly adequate for the unit tests that exercise this module.
"""

from dataclasses import dataclass, field
from typing import Iterable, Sequence, List

import numpy as np

__all__ = ["TransmissionLineSegment", "TriggeredSwitch", "assemble_matrices"]


# ---------------------------------------------------------------------------
# Utility helpers

def _interp_profile(profile: Sequence[tuple[float, float]] | None, t: float) -> float:
    """Return interpolated profile value at time ``t``.

    Profiles are specified as ``[(time, value), ...]`` and are linearly
    interpolated.  If no profile is supplied the return value is ``0.0``.  The
    profile is considered to contain absolute contributions to the quantity in
    question (i.e. they are *added* to the base value).
    """

    if not profile:
        return 0.0
    arr = np.asarray(profile, dtype=float)
    times = arr[:, 0]
    values = arr[:, 1]
    return float(np.interp(t, times, values, left=values[0], right=values[-1]))


# ---------------------------------------------------------------------------
# Component definitions


@dataclass
class TransmissionLineSegment:
    """Simple RLC transmission line segment with optional parasitics.

    Parameters
    ----------
    from_node, to_node:
        Identifiers of the nodes connected by this segment.  The topology is
        not used directly in the tests but is parsed to mirror the behaviour of
        the real application.
    length:
        Physical length of the segment in metres.
    L_per_m, R_per_m, C_per_m:
        Inductance, resistance and capacitance per metre.
    L_parasitic, R_parasitic, C_parasitic:
        Fixed parasitic components attached to the whole segment.
    L_profile, R_profile, C_profile:
        Optional time dependant adjustments (absolute values) for the
        respective quantities.  Each profile is a list of ``(time, value)``
        pairs in SI units.
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

    def totals(self, t: float = 0.0) -> tuple[float, float, float]:
        """Return the total ``(L, R, C)`` for this segment at time ``t``."""

        L = self.L_per_m * self.length + self.L_parasitic + _interp_profile(self.L_profile, t)
        R = self.R_per_m * self.length + self.R_parasitic + _interp_profile(self.R_profile, t)
        C = self.C_per_m * self.length + self.C_parasitic + _interp_profile(self.C_profile, t)
        return L, R, C


@dataclass
class TriggeredSwitch:
    """Ideal resistive switch with optional trigger times and parasitics."""

    from_node: int
    to_node: int
    closed: bool = True
    R_on: float = 1e-3
    R_off: float = 1e6
    trigger_times: Sequence[float] | None = None
    trigger_time: float | None = None  # backward compatible single trigger
    L_parasitic: float = 0.0
    R_parasitic: float = 0.0
    C_parasitic: float = 0.0
    _next_trigger: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # Allow ``trigger_time`` alias for a single entry
        if self.trigger_times is None:
            if self.trigger_time is not None:
                self.trigger_times = [self.trigger_time]
            else:
                self.trigger_times = []
        else:
            self.trigger_times = list(self.trigger_times)
            if self.trigger_time is not None:
                self.trigger_times.append(self.trigger_time)
        # Ensure times are in ascending order for efficient processing
        self.trigger_times = sorted(self.trigger_times)

    # ------------------------------------------------------------------
    def update(self, t: float) -> None:
        """Update the switch state based on ``t``.

        Every time a trigger time is crossed the state is toggled.  The update
        is idempotent: calling it multiple times with the same ``t`` has no
        effect.
        """

        while self._next_trigger < len(self.trigger_times) and t >= self.trigger_times[self._next_trigger]:
            self.closed = not self.closed
            self._next_trigger += 1

    # ------------------------------------------------------------------
    def resistance(self, t: float | None = None) -> float:
        """Return instantaneous resistance including parasitic component."""

        if t is not None:
            self.update(t)
        base = self.R_on if self.closed else self.R_off
        return base + self.R_parasitic


# ---------------------------------------------------------------------------
# Matrix assembly


def assemble_matrices(
    segments: Sequence[TransmissionLineSegment],
    switches: Sequence[TriggeredSwitch] | None = None,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble diagonal ``R``, ``L`` and ``C`` matrices for a network.

    The matrices simply contain the total values of each element on their
    diagonals.  They are primarily a convenience for the unit tests and mimic a
    very small subset of the behaviour of the real solver where a more complex
    topology would be handled.
    """

    R_list: List[float] = []
    L_list: List[float] = []
    C_list: List[float] = []

    for seg in segments:
        L, R, C = seg.totals(t)
        R_list.append(R)
        L_list.append(L)
        C_list.append(C)

    if switches:
        for idx, sw in enumerate(switches):
            if idx < len(R_list):
                R_list[idx] += sw.resistance(t)
                L_list[idx] += sw.L_parasitic
                C_list[idx] += sw.C_parasitic
            else:
                R_list.append(sw.resistance(t))
                L_list.append(sw.L_parasitic)
                C_list.append(sw.C_parasitic)

    n = len(R_list)
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))

    R = np.zeros((n, n))
    L = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(n):
        R[i][i] = R_list[i]
        L[i][i] = L_list[i]
        C[i][i] = C_list[i]
    return R, L, C
