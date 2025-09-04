from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .paschen import paschen_breakdown_time

mu0 = 4e-7 * np.pi


@dataclass
class PrePulseResult:
    time: np.ndarray
    jxb_force: np.ndarray
    breakdown_time: float
    breakdown_index: int


class PrePulseBreakdownModel:
    """Minimal pre-pulse breakdown model with :math:`J\times B` force."""

    def __init__(
        self,
        area: float,
        mass: float,
        force_threshold: float,
        *,
        gap: Optional[float] = None,
        pressure: Optional[float] = None,
        voltage: Optional[float] = None,
    ) -> None:
        self.area = area
        self.mass = mass
        self.force_threshold = force_threshold
        self.radius = np.sqrt(area / np.pi)
        self.gap = gap
        self.pressure = pressure
        self.voltage = voltage

    def run(self, time: Iterable[float], current: Iterable[float]) -> PrePulseResult:
        t = np.array(list(time))
        I = np.array(list(current))
        J = I / self.area
        B = mu0 * I / (2 * np.pi * self.radius)
        jxb = J * B
        idx_candidates = [i for i, val in enumerate(jxb) if val >= self.force_threshold]
        idx_jxb = idx_candidates[0] if idx_candidates else len(t) - 1

        idx_paschen: Optional[int] = None
        if None not in (self.gap, self.pressure, self.voltage):
            t_paschen = paschen_breakdown_time(self.gap, self.pressure, self.voltage)
            idx_paschen = next((i for i, tt in enumerate(t) if tt >= t_paschen), len(t) - 1)

        candidates = [i for i in (idx_jxb, idx_paschen) if i is not None]
        idx = min(candidates)
        return PrePulseResult(time=t, jxb_force=jxb, breakdown_time=float(t[idx]), breakdown_index=idx)
