from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
try:  # pragma: no cover - allow running without SciPy
    from scipy.integrate import solve_ivp
except Exception:  # pragma: no cover
    def solve_ivp(fun, t_span, y0, t_eval, args=(), method="RK45"):
        y = np.zeros((len(y0), len(t_eval)))
        y[:, 0] = y0
        for i in range(len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            y[:, i + 1] = y[:, i] + dt * np.asarray(fun(t_eval[i], y[:, i], *args))
        class _Sol:  # minimal result object
            def __init__(self, y):
                self.y = y
        return _Sol(y)

from .eos import RealGasEOS
from .fusion import bosch_hale_dd
from .hall_mhd_solver import HallMHDSolver, MHDState
from .physics import PicDriver

__all__ = [
    "PinchModelBase",
    "PinchResult",
    "AnalyticPinchModel",
    "SemiAnalyticPinchModel",
    "MHDPinchModel",
    "HybridPinchModel",
]


@dataclass
class PinchResult:
    time: np.ndarray
    radius: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    neutron_yield: float
    axial_position: np.ndarray | None = None
    energy: np.ndarray | None = None


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
        gamma: float = 1.4,
        zeff: float = 1.0,
    ) -> None:
        self.initial_radius = initial_radius
        self.initial_axial = initial_axial
        self.mass = mass
        self.ext_pressure = ext_pressure
        self.damping = damping
        self.eos = RealGasEOS(gamma=gamma)
        self.zeff = zeff

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
        volume = np.pi * r ** 2 * z
        density = self.mass / np.maximum(volume, 1e-12)
        pressure = self.eos.pressure(density, temperature)
        n_i = density / (3.344e-27)  # deuterium ions per m^3
        reactivity = bosch_hale_dd(temperature / 1e3)
        rate = 0.25 * n_i ** 2 * reactivity * volume
        neutron_yield = float(np.trapz(rate, t))
        return PinchResult(t, r, temperature, pressure, neutron_yield, axial_position=z)


class MHDPinchModel(PinchModelBase):
    """Pinch model driven by the simplified Hall-MHD solver."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int] = (8, 8, 8),
        init_density: float = 1.0,
        init_pressure: float = 1e5,
        current_norm: float = 1e4,
    ) -> None:
        self.grid_shape = grid_shape
        self.init_density = init_density
        self.init_pressure = init_pressure
        self.current_norm = current_norm
        nx, ny, nz = grid_shape
        x = np.arange(nx) - nx / 2
        y = np.arange(ny) - ny / 2
        z = np.arange(nz) - nz / 2
        X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
        self.r2 = X**2 + Y**2
        self.volume = float(nx * ny * nz)

    def run(self, time: Iterable[float], current: Iterable[float]) -> PinchResult:
        t = np.asarray(time)
        I = np.asarray(current)
        gamma = 5.0 / 3.0
        solver = HallMHDSolver()

        rho = np.full(self.grid_shape, self.init_density)
        mom = np.zeros(self.grid_shape + (3,))
        B_pattern = np.zeros(self.grid_shape + (3,))
        B_pattern[..., 2] = 1.0
        B = B_pattern * (I[0] / self.current_norm)
        p0 = np.full(self.grid_shape, self.init_pressure)
        internal = p0 / (gamma - 1.0)
        energy = internal + 0.5 * np.sum(B**2, axis=-1)
        state = MHDState(rho=rho, mom=mom, energy=energy, B=B)

        def diagnostics(s: MHDState) -> tuple[float, float, float, float]:
            v = s.mom / s.rho[..., None]
            kinetic = 0.5 * s.rho * np.sum(v**2, axis=-1)
            magnetic = 0.5 * np.sum(s.B**2, axis=-1)
            internal = s.energy - kinetic - magnetic
            p = (gamma - 1.0) * internal
            T = p / s.rho
            rad = np.sqrt(np.sum(s.rho * self.r2) / np.sum(s.rho))
            return rad, float(np.mean(T)), float(np.mean(p)), float(np.sum(s.energy))

        radius = []
        temperature = []
        pressure = []
        energy_hist = []

        rad, temp, pres, Etot = diagnostics(state)
        radius.append(rad)
        temperature.append(temp)
        pressure.append(pres)
        energy_hist.append(Etot)

        neutron_yield = 0.0
        for k in range(len(t) - 1):
            dt = t[k + 1] - t[k]
            state = solver.step(state, dt)
            rad, temp, pres, Etot = diagnostics(state)
            radius.append(rad)
            temperature.append(temp)
            pressure.append(pres)
            energy_hist.append(Etot)
            n_i = np.mean(state.rho) / (3.344e-27)
            reactivity = bosch_hale_dd(max(temp, 0.0) / 1e3)
            mag = np.mean(np.sum(state.B**2, axis=-1))
            rate = 0.25 * n_i**2 * reactivity * mag * self.volume
            neutron_yield += rate * dt

        return PinchResult(
            time=t,
            radius=np.asarray(radius),
            temperature=np.asarray(temperature),
            pressure=np.asarray(pressure),
            neutron_yield=float(neutron_yield),
            energy=np.asarray(energy_hist),
        )


class HybridPinchModel(PinchModelBase):
    """Hybrid pinch model that swaps regions between PIC and fluid solvers.

    The outer plasma is evolved with the simplified Hall‑MHD solver while an
    inner region may be handled by an external PIC driver once the radius falls
    below ``switch_radius``.  When the plasma expands past this radius the model
    reverts to the fluid description.
    """

    def __init__(
        self,
        pic_driver: PicDriver,
        grid_shape: tuple[int, int, int] = (8, 8, 8),
        init_density: float = 1.0,
        init_pressure: float = 1e5,
        current_norm: float = 1e4,
        switch_radius: float = 5e-3,
    ) -> None:
        self.pic_driver = pic_driver
        self.grid_shape = grid_shape
        self.init_density = init_density
        self.init_pressure = init_pressure
        self.current_norm = current_norm
        self.switch_radius = switch_radius
        nx, ny, nz = grid_shape
        x = np.arange(nx) - nx / 2
        y = np.arange(ny) - ny / 2
        z = np.arange(nz) - nz / 2
        X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
        self.r2 = X**2 + Y**2
        self.volume = float(nx * ny * nz)

    def run(self, time: Iterable[float], current: Iterable[float]) -> PinchResult:
        t = np.asarray(time)
        I = np.asarray(current)
        gamma = 5.0 / 3.0
        solver = HallMHDSolver()

        rho = np.full(self.grid_shape, self.init_density)
        mom = np.zeros(self.grid_shape + (3,))
        B_pattern = np.zeros(self.grid_shape + (3,))
        B_pattern[..., 2] = 1.0
        B = B_pattern * (I[0] / self.current_norm)
        p0 = np.full(self.grid_shape, self.init_pressure)
        internal = p0 / (gamma - 1.0)
        energy = internal + 0.5 * np.sum(B**2, axis=-1)
        state = MHDState(rho=rho, mom=mom, energy=energy, B=B)

        def diagnostics(s: MHDState) -> tuple[float, float, float, float]:
            v = s.mom / s.rho[..., None]
            kinetic = 0.5 * s.rho * np.sum(v**2, axis=-1)
            magnetic = 0.5 * np.sum(s.B**2, axis=-1)
            internal = s.energy - kinetic - magnetic
            p = (gamma - 1.0) * internal
            T = p / s.rho
            rad = np.sqrt(np.sum(s.rho * self.r2) / np.sum(s.rho))
            return rad, float(np.mean(T)), float(np.mean(p)), float(np.sum(s.energy))

        radius: list[float] = []
        temperature: list[float] = []
        pressure: list[float] = []
        energy_hist: list[float] = []

        rad, temp, pres, Etot = diagnostics(state)
        use_pic = rad <= self.switch_radius
        if use_pic:
            rad_pic, e_pic = self.pic_driver.step(I[0], 0.0)
            rad = rad_pic
        else:
            e_pic = 0.0
        radius.append(rad)
        temperature.append(temp)
        pressure.append(pres)
        energy_hist.append(Etot + e_pic)

        neutron_yield = 0.0
        for k in range(len(t) - 1):
            dt = t[k + 1] - t[k]
            state = solver.step(state, dt, current=I[k])
            rad_fluid, temp, pres, Etot = diagnostics(state)
            if use_pic or rad_fluid <= self.switch_radius:
                rad_pic, e_pic = self.pic_driver.step(I[k], dt if use_pic else 0.0)
                rad = rad_pic
                use_pic = True
                if rad_fluid > self.switch_radius:
                    use_pic = False
            else:
                rad = rad_fluid
                e_pic = 0.0
                if rad_fluid <= self.switch_radius:
                    rad_pic, e_pic = self.pic_driver.step(I[k], 0.0)
                    rad = rad_pic
                    use_pic = True
            radius.append(rad)
            temperature.append(temp)
            pressure.append(pres)
            energy_hist.append(Etot + e_pic)
            n_i = np.mean(state.rho) / (3.344e-27)
            reactivity = bosch_hale_dd(max(temp, 0.0) / 1e3)
            rate = 0.25 * n_i**2 * reactivity * self.volume
            neutron_yield += rate * dt

        return PinchResult(
            time=t,
            radius=np.asarray(radius),
            temperature=np.asarray(temperature),
            pressure=np.asarray(pressure),
            neutron_yield=float(neutron_yield),
            energy=np.asarray(energy_hist),
        )

