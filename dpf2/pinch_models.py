from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["PinchModelBase", "PinchResult", "AnalyticPinchModel", "SemiAnalyticPinchModel"]


@dataclass
class PinchResult:
    time: np.ndarray
    radius: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    neutron_yield: float
    axial_position: np.ndarray | None = None


class PinchModelBase:
    """Base interface for pinch dynamics models."""

    def run(self, time: Iterable[float], current: Iterable[float]) -> PinchResult:  # pragma: no cover - interface
        raise NotImplementedError


class AnalyticPinchModel(PinchModelBase):
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


class SemiAnalyticPinchModel(PinchModelBase):
    """Cylindrical collapse model with simple pressure balance."""

    def __init__(
        self,
        initial_radius: float = 1e-2,
        initial_axial: float = 0.1,
        mass: float = 1e-6,
        ext_pressure: float = 1e5,
        damping: float = 0.0,
    ) -> None:
        self.initial_radius = initial_radius
        self.initial_axial = initial_axial
        self.mass = mass
        self.ext_pressure = ext_pressure
        self.damping = damping

    def _dynamics(self, t: float, y: np.ndarray, current: np.ndarray, time: np.ndarray) -> np.ndarray:
        r, vr, z, vz = y
        I = np.interp(t, time, current)
        # magnetic pressure term; avoid divide by zero
        force_r = (1e-7 * I ** 2) / max(r, 1e-6)  # approx mu0/(2*pi)=2e-7, simplified
        acc_r = (force_r - self.ext_pressure * r) / self.mass - self.damping * vr
        acc_z = -self.ext_pressure / self.mass - self.damping * vz
        return np.array([vr, acc_r, vz, acc_z])

    def run(self, time: Iterable[float], current: Iterable[float]) -> PinchResult:
        t = np.asarray(time)
        I = np.asarray(current)
        y0 = [self.initial_radius, 0.0, self.initial_axial, 0.0]
        sol = solve_ivp(self._dynamics, (t[0], t[-1]), y0, t_eval=t, args=(I, t), method="RK45")
        r = sol.y[0]
        z = sol.y[2]
        temperature = 1e3 * (I / 1e4) ** 2 + 0.1 * r ** -1
        pressure = self.ext_pressure + 0.5 * (I ** 2) * 1e-6
        yield_integrand = (temperature / 1e3) ** 3 * I ** 2
        neutron_yield = float(np.trapz(yield_integrand, t) * 1e-20)
        return PinchResult(t, r, temperature, pressure, neutron_yield, axial_position=z)

