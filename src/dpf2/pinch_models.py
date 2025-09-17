from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

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
from .physics.inductance import (
    CoaxialGeometry,
    dynamic_inductance,
    dynamic_inductance_with_derivatives,
)
from .diagnostics import (
    plasma_beta,
    alfven_mach_number,
    magnetic_reynolds_number,
    lundquist_number,
)

try:  # pragma: no cover - optional constants
    from scipy.constants import mu_0, k as k_B, e as q_e, m_p
except Exception:  # pragma: no cover
    mu_0 = 4e-7 * np.pi
    k_B = 1.380649e-23
    q_e = 1.602176634e-19
    m_p = 1.67262192369e-27

__all__ = [
    "PinchModelBase",
    "PinchResult",
    "AnalyticPinchModel",
    "SemiAnalyticPinchModel",
    "MHDPinchModel",
    "HybridPinchModel",
    "SnowplowPinchModel",
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
    beta: np.ndarray | None = None
    alfven_mach: np.ndarray | None = None
    magnetic_reynolds: np.ndarray | None = None
    lundquist: np.ndarray | None = None
    voltage: np.ndarray | None = None
    current: np.ndarray | None = None
    rayleigh_taylor_growth: np.ndarray | None = None


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
        B = 4e-7 * np.pi * I / (2 * np.pi * radius)
        n = pressure / np.maximum(temperature, 1e-12) / 1.380649e-23
        beta = np.array([plasma_beta(ni, Ti, Bi) for ni, Ti, Bi in zip(n, temperature, B)])
        vr = np.gradient(radius, t, edge_order=2)
        alfven = np.array([alfven_mach_number(v, Bi, ni) for v, Bi, ni in zip(vr, B, n)])
        sigma = 1e5
        rm = np.array([magnetic_reynolds_number(abs(v), r, sigma) for v, r in zip(vr, radius)])
        s = np.array([lundquist_number(Bi, ni, r, sigma) for Bi, ni, r in zip(B, n, radius)])
        return PinchResult(
            t,
            radius,
            temperature,
            pressure,
            neutron_yield,
            beta=beta,
            alfven_mach=alfven,
            magnetic_reynolds=rm,
            lundquist=s,
        )


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
        B = 4e-7 * np.pi * I / (2 * np.pi * r)
        beta = np.array([plasma_beta(ni, Ti, Bi) for ni, Ti, Bi in zip(n_i, temperature, B)])
        vr = sol.y[1]
        alfven = np.array([alfven_mach_number(v, Bi, ni) for v, Bi, ni in zip(vr, B, n_i)])
        sigma = 1e5
        rm = np.array([magnetic_reynolds_number(abs(v), rad, sigma) for v, rad in zip(vr, r)])
        s = np.array([lundquist_number(Bi, ni, rad, sigma) for Bi, ni, rad in zip(B, n_i, r)])
        return PinchResult(
            t,
            r,
            temperature,
            pressure,
            neutron_yield,
            axial_position=z,
            beta=beta,
            alfven_mach=alfven,
            magnetic_reynolds=rm,
            lundquist=s,
        )


class SnowplowPinchModel(PinchModelBase):
    """Two-stage slug model with Rayleigh-Taylor diagnostics."""

    def __init__(
        self,
        geometry: CoaxialGeometry | None = None,
        fill_pressure: float = 1.5e3,
        fill_temperature: float = 300.0,
        species_mass: float = 2.0 * m_p,
        mode_number: int = 12,
        resistivity: float = 5e-6,
    ) -> None:
        self.geometry = geometry or CoaxialGeometry(
            anode_radius=8.0e-3, cathode_radius=4.0e-2, anode_length=0.18
        )
        self.fill_pressure = fill_pressure
        self.fill_temperature = fill_temperature
        self.species_mass = species_mass
        self.mode_number = mode_number
        self.resistivity = resistivity
        self._annulus_area = np.pi * (
            self.geometry.cathode_radius**2 - self.geometry.anode_radius**2
        )
        self._rho0 = (
            fill_pressure * species_mass / (k_B * max(fill_temperature, 1.0))
        )
        self._mass_floor = 1e-9

    def run(self, time: Iterable[float], current: Iterable[float]) -> PinchResult:
        t = np.asarray(time)
        I = np.asarray(current)
        if t.ndim != 1 or I.ndim != 1 or len(t) != len(I):
            raise ValueError("time and current must be one-dimensional arrays of equal length")

        geom = self.geometry
        radius = np.zeros_like(t)
        axial = np.zeros_like(t)
        temperature = np.zeros_like(t)
        pressure = np.zeros_like(t)
        beta_hist = np.zeros_like(t)
        alfven_hist = np.zeros_like(t)
        rm_hist = np.zeros_like(t)
        s_hist = np.zeros_like(t)
        energy_hist = np.zeros_like(t)
        rt_hist = np.zeros_like(t)
        voltage_hist = np.zeros_like(t)

        r = geom.cathode_radius
        z = 0.0
        vr = 0.0
        vz = 0.0
        mass = self._mass_floor
        stage = "axial"
        neutron_yield = 0.0
        rt_integral = 0.0

        for k, tk in enumerate(t):
            I_k = float(I[k])
            r_eff = max(r, 0.5 * geom.anode_radius)
            volume = max(np.pi * r_eff**2 * geom.pinch_span, 1e-12)
            density = mass / volume
            n_i = max(density / self.species_mass, 1e6)
            B = mu_0 * I_k / (2.0 * np.pi * max(r_eff, 1e-4))
            magnetic_pressure = B**2 / (2.0 * mu_0)
            dynamic_pressure = 0.5 * density * (vr**2 + vz**2)
            pressure_k = magnetic_pressure + dynamic_pressure + self.fill_pressure
            T_K = max(pressure_k / (n_i * k_B), 1.0)
            T_keV = T_K * k_B / (1e3 * q_e)
            reactivity = bosch_hale_dd(max(T_keV, 1e-3))
            rate = 0.25 * n_i**2 * reactivity
            if k + 1 < len(t):
                dt = t[k + 1] - tk
            else:
                dt = (t[k] - t[k - 1]) if k > 0 else 0.0
            neutron_yield += rate * volume * max(dt, 0.0)

            Lp, dL_dz, dL_dr = dynamic_inductance_with_derivatives(z, r_eff, geom)
            dL_dt = dL_dz * vz + dL_dr * vr
            voltage_hist[k] = -I_k * dL_dt
            energy_hist[k] = 0.5 * Lp * I_k**2

            radius[k] = r
            axial[k] = z
            temperature[k] = T_K
            pressure[k] = pressure_k
            beta_hist[k] = plasma_beta(n_i, T_K, B)
            v_char = float(np.sqrt(vr**2 + vz**2))
            alfven_hist[k] = alfven_mach_number(v_char, B, n_i)
            sigma = 1.0 / max(self.resistivity, 1e-8)
            rm_hist[k] = magnetic_reynolds_number(max(abs(v_char), 1e-9), r_eff, sigma)
            s_hist[k] = lundquist_number(B, n_i, r_eff, sigma)
            rt_hist[k] = rt_integral

            if k == len(t) - 1:
                continue

            dt = t[k + 1] - tk
            if dt <= 0.0:
                continue

            if stage == "axial" and z < geom.anode_length:
                dm_dt = self._rho0 * self._annulus_area * max(vz, 0.0)
                F_mag = 0.5 * I_k**2 * geom.axial_gradient
                F_back = self.fill_pressure * self._annulus_area
                accel = (F_mag - F_back - vz * dm_dt) / max(mass, self._mass_floor)
                vz_new = vz + accel * dt
                z += 0.5 * (vz + vz_new) * dt
                mass += dm_dt * dt
                vz = vz_new
                if z >= geom.anode_length:
                    z = geom.anode_length
                    stage = "radial"
                    vz = 0.0
            else:
                surface = 2.0 * np.pi * r_eff * geom.pinch_span
                F_mag = 0.5 * I_k**2 * (-geom.radial_gradient_scale / r_eff)
                F_back = self.fill_pressure * surface
                accel = (F_mag - F_back) / max(mass, self._mass_floor)
                vr_new = vr + accel * dt
                r += 0.5 * (vr + vr_new) * dt
                r = max(r, 0.4 * geom.anode_radius)
                if r <= 0.4 * geom.anode_radius:
                    vr_new = 0.0
                vr = vr_new
                if accel < 0.0:
                    k_mode = self.mode_number / max(r, geom.anode_radius * 0.4)
                    gamma = np.sqrt(abs(accel) * k_mode)
                    rt_integral += gamma * dt

        return PinchResult(
            t,
            radius,
            temperature,
            pressure,
            float(neutron_yield),
            axial_position=axial,
            energy=energy_hist,
            beta=beta_hist,
            alfven_mach=alfven_hist,
            magnetic_reynolds=rm_hist,
            lundquist=s_hist,
            voltage=voltage_hist,
            current=I,
            rayleigh_taylor_growth=rt_hist,
        )


class MHDPinchModel(PinchModelBase):
    """Pinch model driven by the simplified Hall-MHD solver."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int] = (8, 8, 8),
        init_density: float = 1.0,
        init_pressure: float = 1e5,
        current_norm: float = 1e4,
        resistivity_model: Any | None = None,
    ) -> None:
        self.grid_shape = grid_shape
        self.init_density = init_density
        self.init_pressure = init_pressure
        self.current_norm = current_norm
        self.resistivity_model = resistivity_model
        nx, ny, nz = grid_shape
        x = (np.arange(nx) - nx / 2) * 1e-3
        y = (np.arange(ny) - ny / 2) * 1e-3
        z = (np.arange(nz) - nz / 2) * 1e-3
        X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
        self.r2 = X**2 + Y**2
        self.volume = float(nx * ny * nz)

    def run(self, time: Iterable[float], current: Iterable[float]) -> PinchResult:
        t = np.asarray(time)
        I = np.asarray(current).copy()
        gamma = 5.0 / 3.0
        solver = HallMHDSolver(
            anomalous_resistivity=(
                self.resistivity_model.anomalous_resistivity
                if self.resistivity_model is not None
                else None
            )
        )

        rho = np.full(self.grid_shape, self.init_density)
        mom = np.zeros(self.grid_shape + (3,))
        B_pattern = np.zeros(self.grid_shape + (3,))
        B_pattern[..., 2] = 1.0
        B = B_pattern * (I[0] / self.current_norm)
        p0 = np.full(self.grid_shape, self.init_pressure)
        internal = p0 / (gamma - 1.0)
        energy = internal + 0.5 * np.sum(B**2, axis=-1)
        state = MHDState(rho=rho, mom=mom, energy=energy, B=B)

        def diagnostics(s: MHDState) -> tuple[float, float, float, float, float, float, float, float]:
            v = s.mom / s.rho[..., None]
            vmag = np.sqrt(np.sum(v**2, axis=-1))
            kinetic = 0.5 * s.rho * vmag**2
            Bmag = np.sqrt(np.sum(s.B**2, axis=-1))
            magnetic = 0.5 * Bmag**2
            internal = s.energy - kinetic - magnetic
            p = (gamma - 1.0) * internal
            T = p / s.rho
            rad = np.sqrt(np.sum(s.rho * self.r2) / np.sum(s.rho))
            n = s.rho / (3.344e-27)
            n_mean = float(np.mean(n))
            B_mean = float(np.mean(Bmag))
            beta = plasma_beta(n_mean, float(np.mean(T)), B_mean)
            v_mean = float(np.mean(vmag))
            alfven = alfven_mach_number(v_mean, B_mean, n_mean)
            sigma = 1e5
            rm = magnetic_reynolds_number(v_mean, rad, sigma)
            s_num = lundquist_number(B_mean, n_mean, rad, sigma)
            return rad, float(np.mean(T)), float(np.mean(p)), float(np.sum(s.energy)), beta, alfven, rm, s_num

        radius = []
        temperature = []
        pressure = []
        energy_hist = []
        beta_hist = []
        alfven_hist = []
        rm_hist = []
        s_hist = []

        rad, temp, pres, Etot, b, a, rm, s_num = diagnostics(state)
        radius.append(rad)
        temperature.append(temp)
        pressure.append(pres)
        energy_hist.append(Etot)
        beta_hist.append(b)
        alfven_hist.append(a)
        rm_hist.append(rm)
        s_hist.append(s_num)
        voltage_hist: list[float] = [0.0]

        neutron_yield = 0.0
        k_field = None
        if self.resistivity_model is not None:
            if self.resistivity_model.amplitude is None:
                self.resistivity_model.amplitude = np.zeros(self.grid_shape)
            k_field = np.zeros(self.grid_shape) + 0.1
        for k in range(len(t) - 1):
            dt = t[k + 1] - t[k]
            if self.resistivity_model is not None and k_field is not None:
                self.resistivity_model.evolve(
                    self.resistivity_model.amplitude, k_field, dt
                )
            state = solver.step(state, dt, current=float(I[k]))
            rad, temp, pres, Etot, b, a, rm, s_num = diagnostics(state)
            radius.append(rad)
            temperature.append(temp)
            pressure.append(pres)
            energy_hist.append(Etot)
            beta_hist.append(b)
            alfven_hist.append(a)
            rm_hist.append(rm)
            s_hist.append(s_num)
            spike = solver.voltage_spikes[-1] if solver.voltage_spikes else 0.0
            voltage_hist.append(spike)
            if k + 1 < len(I):
                I[k + 1] = max(I[k + 1] - spike * dt / self.current_norm, 0.0)
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
            beta=np.asarray(beta_hist),
            alfven_mach=np.asarray(alfven_hist),
            magnetic_reynolds=np.asarray(rm_hist),
            lundquist=np.asarray(s_hist),
            voltage=np.asarray(voltage_hist),
            current=I,
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
        x = (np.arange(nx) - nx / 2) * 1e-3
        y = (np.arange(ny) - ny / 2) * 1e-3
        z = (np.arange(nz) - nz / 2) * 1e-3
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

        def diagnostics(s: MHDState) -> tuple[float, float, float, float, float, float, float, float]:
            v = s.mom / s.rho[..., None]
            vmag = np.sqrt(np.sum(v**2, axis=-1))
            kinetic = 0.5 * s.rho * vmag**2
            Bmag = np.sqrt(np.sum(s.B**2, axis=-1))
            magnetic = 0.5 * Bmag**2
            internal = s.energy - kinetic - magnetic
            p = (gamma - 1.0) * internal
            T = p / s.rho
            rad = np.sqrt(np.sum(s.rho * self.r2) / np.sum(s.rho))
            n = s.rho / (3.344e-27)
            n_mean = float(np.mean(n))
            B_mean = float(np.mean(Bmag))
            beta = plasma_beta(n_mean, float(np.mean(T)), B_mean)
            v_mean = float(np.mean(vmag))
            alfven = alfven_mach_number(v_mean, B_mean, n_mean)
            sigma = 1e5
            rm = magnetic_reynolds_number(v_mean, rad, sigma)
            s_num = lundquist_number(B_mean, n_mean, rad, sigma)
            return rad, float(np.mean(T)), float(np.mean(p)), float(np.sum(s.energy)), beta, alfven, rm, s_num

        radius: list[float] = []
        temperature: list[float] = []
        pressure: list[float] = []
        energy_hist: list[float] = []
        beta_hist: list[float] = []
        alfven_hist: list[float] = []
        rm_hist: list[float] = []
        s_hist: list[float] = []

        rad_fluid, temp, pres, Etot, b, a, rm, s_num = diagnostics(state)
        rad_pic, e_pic, j_pic = self.pic_driver.step(state, I[0], 0.0)
        getattr(self.pic_driver, "exchange_fields", lambda: (None, None))()
        getattr(self.pic_driver, "exchange_particles", lambda: (None, None))()
        use_pic = rad_fluid <= self.switch_radius
        rad = rad_pic if use_pic else rad_fluid
        radius.append(rad)
        temperature.append(temp)
        pressure.append(pres)
        energy_hist.append(Etot + e_pic)
        beta_hist.append(b)
        alfven_hist.append(a)
        rm_hist.append(rm)
        s_hist.append(s_num)

        neutron_yield = 0.0
        for k in range(len(t) - 1):
            dt = t[k + 1] - t[k]
            rad_pic, e_pic, j_pic = self.pic_driver.step(state, I[k], dt)
            getattr(self.pic_driver, "exchange_fields", lambda: (None, None))()
            getattr(self.pic_driver, "exchange_particles", lambda: (None, None))()
            state = solver.step(state, dt, current=j_pic)
            rad_fluid, temp, pres, Etot, b, a, rm, s_num = diagnostics(state)
            if use_pic:
                if rad_fluid > self.switch_radius:
                    use_pic = False
            else:
                if rad_fluid <= self.switch_radius:
                    use_pic = True
            rad = rad_pic if use_pic else rad_fluid
            radius.append(rad)
            temperature.append(temp)
            pressure.append(pres)
            energy_hist.append(Etot + e_pic)
            beta_hist.append(b)
            alfven_hist.append(a)
            rm_hist.append(rm)
            s_hist.append(s_num)
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
            beta=np.asarray(beta_hist),
            alfven_mach=np.asarray(alfven_hist),
            magnetic_reynolds=np.asarray(rm_hist),
            lundquist=np.asarray(s_hist),
        )

