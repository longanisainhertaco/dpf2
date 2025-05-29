"""Simple 0D analytic pinch model for DPF simulations."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import numpy as np

__all__ = ["AnalyticPinchModel", "PinchResult"]


@dataclass
class PinchResult:
    time: np.ndarray
    radius: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    neutron_yield: float


class AnalyticPinchModel:
    """Very simple analytic model of the DPF pinch."""

    def __init__(self, initial_radius: float = 1e-2, tau: float = 50e-9) -> None:
        self.initial_radius = initial_radius
        self.tau = tau

    def run(self, time: Iterable[float], current: Iterable[float]) -> PinchResult:
        t = np.asarray(time)
        I = np.asarray(current)
        radius = self.initial_radius * np.exp(-t / self.tau)
        pressure = 0.5 * (I ** 2) * 1e-6  # arbitrary scaling
        temperature = 1e3 * (I / 1e4) ** 2
        yield_integrand = (temperature / 1e3) ** 3 * I ** 2
        neutron_yield = float(np.trapz(yield_integrand, t) * 1e-20)
        return PinchResult(t, radius, temperature, pressure, neutron_yield)
