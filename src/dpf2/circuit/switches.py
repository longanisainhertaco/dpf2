from __future__ import annotations

"""Switch models for the distributed circuit solver.

This module exposes a minimal subset of the behaviour of the real project in a
form that is sufficient for the unit tests.  Two switch types are provided:

``TriggeredSwitch``
    Generic resistive switch which can change its state at specified trigger
    times.  Each trigger time may be subjected to optional Gaussian jitter.  The
    switch can also model arc growth by increasing its on--resistance linearly
    with time after closure.

``CrowbarStage``
    Convenience wrapper representing a crowbar branch that engages at a given
    trigger time with a fixed resistance.

The classes here mirror the implementations previously embedded in
``circuit.distributed`` but are factored out to keep the solver logic focused on
network assembly.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import random

__all__ = ["TriggeredSwitch", "CrowbarStage"]


# ---------------------------------------------------------------------------
# Base switch model


@dataclass
class TriggeredSwitch:
    """Ideal resistive switch with optional trigger times and parasitics.

    Parameters
    ----------
    from_node, to_node:
        Node identifiers the switch connects.
    closed:
        Initial state of the switch.
    R_on, R_off:
        Resistance when the switch is closed or open respectively.
    trigger_times:
        Sequence of times (in seconds) at which the state toggles.  Gaussian
        jitter with standard deviation ``jitter_std`` may be applied to each
        trigger time.
    arc_resistance:
        Optional linear increase of the on--resistance after the switch has
        closed to emulate arc growth.  The value represents Ohms per second.
    L_parasitic, R_parasitic, C_parasitic:
        Optional lumped parasitic components associated with the switch.
    """

    from_node: int
    to_node: int
    closed: bool = True
    R_on: float = 1e-3
    R_off: float = 1e6
    trigger_times: Sequence[float] | None = None
    trigger_time: float | None = None  # backward compatible single trigger
    jitter_std: float = 0.0
    arc_resistance: float = 0.0
    L_parasitic: float = 0.0
    R_parasitic: float = 0.0
    C_parasitic: float = 0.0

    _next_trigger: int = field(init=False, default=0)
    _closed_since: float | None = field(init=False, default=None)

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
        # Apply optional Gaussian jitter
        if self.jitter_std > 0.0 and self.trigger_times:
            try:
                jitter = np.random.normal(0.0, self.jitter_std, len(self.trigger_times))  # type: ignore[arg-type]
            except Exception:
                jitter = [random.gauss(0.0, self.jitter_std) for _ in self.trigger_times]
            self.trigger_times = [t + j for t, j in zip(self.trigger_times, jitter)]
        self.trigger_times = sorted(self.trigger_times)

    # ------------------------------------------------------------------
    def update(self, t: float) -> None:
        """Update the switch state based on ``t``.

        Every time a trigger time is crossed the switch toggles.  The update is
        idempotent so calling it multiple times with the same ``t`` is safe.
        """

        while self._next_trigger < len(self.trigger_times) and t >= self.trigger_times[self._next_trigger]:
            self.closed = not self.closed
            self._closed_since = t if self.closed else None
            self._next_trigger += 1

    # ------------------------------------------------------------------
    def resistance(self, t: float | None = None) -> float:
        """Return instantaneous resistance including parasitic component."""

        if t is not None:
            self.update(t)
        base = self.R_on if self.closed else self.R_off
        if (
            self.closed
            and self.arc_resistance > 0.0
            and self._closed_since is not None
            and t is not None
        ):
            base += self.arc_resistance * max(0.0, t - self._closed_since)
        return base + self.R_parasitic


# ---------------------------------------------------------------------------
# Crowbar stage convenience wrapper


@dataclass
class CrowbarStage(TriggeredSwitch):
    """Resistive crowbar stage engaging at ``trigger_time`` with optional jitter."""

    def __init__(
        self,
        from_node: int,
        to_node: int,
        resistance: float,
        trigger_time: float,
        jitter_std: float = 0.0,
    ) -> None:
        super().__init__(
            from_node=from_node,
            to_node=to_node,
            closed=False,
            R_on=resistance,
            R_off=1e12,
            trigger_times=[trigger_time],
            jitter_std=jitter_std,
        )
