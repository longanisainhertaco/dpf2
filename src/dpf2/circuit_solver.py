"""Simple RLC circuit solver for DPF simulations.

This module provides a minimal implementation of a series RLC
circuit with constant parameters.  Two integration approaches are
available:

- An analytical solution assuming fixed L, R and C.
- Numerical integration of the ODEs using :func:`scipy.integrate.solve_ivp`.

All quantities are in SI units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .circuit_config import CircuitConfig

__all__ = ["CircuitSolver", "RLCCircuit", "run_circuit_simulation"]


@dataclass
class RLCCircuit:
    """Series RLC circuit parameters."""

    L: float  # Henries
    R: float  # Ohms
    C: float  # Farads
    V0: float  # Volts


class CircuitSolver:
    """Compute current evolution for a series RLC circuit."""

    def __init__(self, circuit: RLCCircuit) -> None:
        self.circuit = circuit
        self.time = [0.0]
        self.currents = [0.0]
        self.voltages = [circuit.V0]

    # ------------------------------------------------------------------
    def _analytical_current(self, t: np.ndarray) -> np.ndarray:
        L, R, C, V0 = (
            self.circuit.L,
            self.circuit.R,
            self.circuit.C,
            self.circuit.V0,
        )
        alpha = R / (2 * L)
        omega0 = 1.0 / np.sqrt(L * C)
        if np.isclose(alpha, omega0):
            # Critically damped
            I = (V0 / L) * t * np.exp(-alpha * t)
        elif alpha > omega0:
            # Over damped
            s1 = -alpha + np.sqrt(alpha**2 - omega0**2)
            s2 = -alpha - np.sqrt(alpha**2 - omega0**2)
            I = (V0 / L) * (np.exp(s1 * t) - np.exp(s2 * t)) / (s1 - s2)
        else:
            # Under damped
            omega_d = np.sqrt(max(0.0, omega0**2 - alpha**2))
            I = (V0 / (L * omega_d)) * np.exp(-alpha * t) * np.sin(omega_d * t)
        return I

    # ------------------------------------------------------------------
    def _numeric_current(self, t: np.ndarray) -> np.ndarray:
        L, R, C, V0 = (
            self.circuit.L,
            self.circuit.R,
            self.circuit.C,
            self.circuit.V0,
        )

        def rhs(t: float, y: Tuple[float, float]):
            Q, I = y
            dQdt = -I
            dIdt = -(R * I + Q / C) / L
            return [dQdt, dIdt]

        y0 = [C * V0, 0.0]
        sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method="RK45")
        return sol.y[1]

    # ------------------------------------------------------------------
    def solve(self, t_end: float, dt: float, method: str = "analytical") -> Tuple[np.ndarray, np.ndarray]:
        """Compute current over ``[0, t_end]``.

        Parameters
        ----------
        t_end:
            Final time in seconds.
        dt:
            Time step in seconds.
        method:
            ``"analytical"`` or ``"ode"`` for numerical integration.
        """
        t = np.arange(0.0, t_end + dt, dt)
        if method == "analytical":
            I = self._analytical_current(t)
        elif method == "ode":
            I = self._numeric_current(t)
        else:
            raise ValueError("method must be 'analytical' or 'ode'")
        return t, I

    def step(
        self,
        current: float,
        back_emf: float,
        dt: float,
        plasma_feedback: dict[str, float] | None = None,
    ) -> Tuple[float, float]:
        """Explicit Euler advance with optional plasma feedback."""

        voltage = self.voltages[-1]
        t = self.time[-1]

        Lp = 0.0
        dLpdt = 0.0
        emf = 0.0
        use_emf = False
        if plasma_feedback:
            Lp = plasma_feedback.get("Lp", 0.0)
            if "emf" in plasma_feedback:
                emf = plasma_feedback["emf"]
                use_emf = True
            else:
                dLpdt = plasma_feedback.get("dLpdt", 0.0)

        Ltot = self.circuit.L + Lp
        if use_emf:
            num = self.circuit.V0 - self.circuit.R * current - voltage - emf - back_emf
        else:
            num = (
                self.circuit.V0
                - self.circuit.R * current
                - voltage
                - dLpdt * current
                - back_emf
            )
        dIdt = num / Ltot
        dVdt = -current / self.circuit.C

        new_current = current + dIdt * dt
        new_voltage = voltage + dVdt * dt

        self.time.append(t + dt)
        self.currents.append(new_current)
        self.voltages.append(new_voltage)

        return new_current, new_voltage


def _profile_to_interp(profile, t_scale: float, y_scale: float):
    """Return interpolation functions for profile value and derivative."""
    if profile is None:
        return (lambda t: 0.0, lambda t: 0.0)
    arr = np.asarray(profile, dtype=float)
    t = arr[:, 0] * t_scale
    y = arr[:, 1] * y_scale
    dy_dt = np.gradient(y, t, edge_order=2)

    def val(tt):
        return np.interp(tt, t, y, left=y[0], right=y[-1])

    def deriv(tt):
        return np.interp(tt, t, dy_dt, left=dy_dt[0], right=dy_dt[-1])

    return val, deriv


def run_circuit_simulation(
    cfg: CircuitConfig, t_end: float, num_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run RLC discharge with optional plasma and mutual inductance.

    Parameters
    ----------
    cfg:
        Circuit configuration including optional coupling profiles.
    t_end:
        End time in microseconds.
    num_points:
        Number of output points, by default ``1000``.

    Returns
    -------
    tuple of ndarray
        time [s], current [A], capacitor voltage [V], mutual current [A],
        mutual-induced voltage [V].
    """

    L_ext = cfg.L_ext * 1e-6
    R = cfg.R_ext * 1e-3
    C = cfg.C_ext * 1e-6
    V0 = cfg.V0 * 1e3
    delay = cfg.switch_delay * 1e-9

    lp_func, dlpdt_func = _profile_to_interp(cfg.plasma_inductance_profile, 1e-6, 1e-6)
    m_func, _ = _profile_to_interp(cfg.mutual_inductance_profile, 1e-6, 1e-6)
    im_func, dim_dt_func = _profile_to_interp(cfg.mutual_current_profile, 1e-6, 1e3)

    def circuit_ode(t, y):
        I, Q = y
        Lp = lp_func(t)
        dLpdt = dlpdt_func(t)
        M = m_func(t)
        dI_mutual_dt = dim_dt_func(t)
        Ltot = L_ext + Lp
        V_mutual = -M * dI_mutual_dt
        dIdt = (V0 + V_mutual - R * I - Q / C - dLpdt * I) / Ltot
        dQdt = I
        return [dIdt, dQdt]

    t_total = np.linspace(0.0, t_end * 1e-6, num_points)

    if t_end * 1e-6 <= delay:
        current = np.zeros_like(t_total)
        voltage = np.full_like(t_total, V0)
        i_mutual = im_func(t_total)
        v_mutual = -m_func(t_total) * dim_dt_func(t_total)
        return t_total, current, voltage, i_mutual, v_mutual

    mask_before = t_total < delay
    current = np.zeros_like(t_total)
    voltage = np.full_like(t_total, V0)
    i_mutual = im_func(t_total)
    v_mutual = -m_func(t_total) * dim_dt_func(t_total)

    t_eval = t_total[~mask_before]
    q0 = C * V0
    sol = solve_ivp(circuit_ode, (delay, t_total[-1]), [0.0, q0], t_eval=t_eval, method="BDF")

    current[~mask_before] = sol.y[0]
    voltage[~mask_before] = sol.y[1] / C

    return t_total, current, voltage, i_mutual, v_mutual
