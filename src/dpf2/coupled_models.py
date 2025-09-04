from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np

from .pinch_models import PinchModelBase, PinchResult
from .prepulse import PrePulseBreakdownModel, PrePulseResult
from .axial_sheath import AxialSheathModel, SheathResult
from .neutral.dsmc import DSMC


@dataclass
class EndToEndResult:
    prepulse: PrePulseResult
    sheath: SheathResult
    pinch: PinchResult


class CoupledEndToEndModel:
    """Run pre-pulse, sheath, and pinch phases sequentially."""

    def __init__(self, pre_pulse: PrePulseBreakdownModel, sheath: AxialSheathModel, pinch: PinchModelBase) -> None:
        self.pre_pulse = pre_pulse
        self.sheath = sheath
        self.pinch = pinch

    def run(self, time: Iterable[float], current: Iterable[float]) -> EndToEndResult:
        t = np.array(list(time))
        I = np.array(list(current))
        pre = self.pre_pulse.run(t, I)
        sheath = self.sheath.run(t, I, start_index=pre.breakdown_index)
        pinch_time = t[sheath.end_index:]
        pinch_current = I[sheath.end_index:]
        pinch = self.pinch.run(pinch_time, pinch_current)
        return EndToEndResult(prepulse=pre, sheath=sheath, pinch=pinch)


@dataclass
class NeutralPlasmaResult:
    """Result container for neutral/plasma coupled runs."""

    neutral_density: float
    plasma: Any


class NeutralPlasmaCoupler:
    """Couple a DSMC neutral solver to an arbitrary plasma solver.

    The plasma solver is expected to provide a ``run`` method accepting a
    ``neutral_density`` keyword argument.  This minimal interface keeps
    the implementation lightweight while remaining sufficiently flexible
    for the unit tests.
    """

    def __init__(self, dsmc: DSMC, plasma_solver: Any) -> None:
        self.dsmc = dsmc
        self.plasma_solver = plasma_solver

    def run(self, dt: float) -> NeutralPlasmaResult:
        nd = self.dsmc.run(dt)
        plasma = self.plasma_solver.run(neutral_density=nd)
        return NeutralPlasmaResult(neutral_density=nd, plasma=plasma)


__all__ = [
    "EndToEndResult",
    "CoupledEndToEndModel",
    "NeutralPlasmaCoupler",
    "NeutralPlasmaResult",
]
