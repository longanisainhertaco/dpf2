from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

mu0 = 4e-7 * np.pi


@dataclass
class PrePulseResult:
    time: np.ndarray
    jxb_force: np.ndarray
    breakdown_time: float
    breakdown_index: int


class PrePulseBreakdownModel:
    """Minimal pre-pulse breakdown model with :math:`J\times B` force."""

    def __init__(self, area: float, mass: float, force_threshold: float) -> None:
        self.area = area
        self.mass = mass
        self.force_threshold = force_threshold
        self.radius = np.sqrt(area / np.pi)

    def run(self, time: Iterable[float], current: Iterable[float]) -> PrePulseResult:
        t = np.array(list(time))
        I = np.array(list(current))
        J = I / self.area
        B = mu0 * I / (2 * np.pi * self.radius)
        jxb = J * B
        idx_candidates = [i for i, val in enumerate(jxb) if val >= self.force_threshold]
        idx = idx_candidates[0] if idx_candidates else len(t) - 1
        return PrePulseResult(time=t, jxb_force=jxb, breakdown_time=float(t[idx]), breakdown_index=idx)
