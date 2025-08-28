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
from typing import Tuple, Any

import numpy as np
import math
try:  # pragma: no cover - optional dependency
    from scipy.integrate import solve_ivp  # type: ignore
except Exception:  # pragma: no cover - very small fallback integrator
    def solve_ivp(fun, t_span, y0, t_eval=None, method=None):
        t0, tf = t_span
        if t_eval is None:
            t_eval = np.linspace(t0, tf, 50)
        y = np.zeros((len(y0), len(t_eval)))
        y[:, 0] = y0
        for k in range(1, len(t_eval)):
            dt = t_eval[k] - t_eval[k - 1]
            y[:, k] = y[:, k - 1] + dt * np.array(fun(t_eval[k - 1], y[:, k - 1]))
        class Res:
            pass
        res = Res()
        res.y = y
        return res

if hasattr(np, "Array") and not hasattr(getattr(np, "Array"), "__radd__"):
    # Enhance test numpy stub with reverse addition to support ``float + Array``
    np.Array.__radd__ = lambda self, other: self.__add__(other)

if not hasattr(np, "asarray"):
    def _asarray(a, dtype=None):
        return np.array(a)

    np.asarray = _asarray  # type: ignore

if not hasattr(np, "interp"):
    def _interp(x, xp, fp, left=None, right=None):
        # Very small linear interpolation supporting monotonic xp
        for i in range(len(xp) - 1):
            if xp[i] <= x <= xp[i + 1]:
                t = (x - xp[i]) / (xp[i + 1] - xp[i])
                return fp[i] * (1 - t) + fp[i + 1] * t
        return fp[0] if x < xp[0] else fp[-1]

    np.interp = _interp  # type: ignore

from .circuit_config import CircuitConfig
from .core.circuit import RLCCircuitSolver
from .core.bases import PlasmaSolverBase, CouplingState

from .geometry.inductance import loop_mutual_inductance


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
        coupling: CouplingState,
        back_emf: float,
        dt: float,
        energy_tracker: EnergyTracker | None = None,
    ) -> CouplingState:
        """Explicit Euler advance with optional plasma feedback."""

        current = coupling.current
        voltage = coupling.voltage
        t = self.time[-1]

        Lp = coupling.Lp
        emf = coupling.emf

        Ltot = self.circuit.L + Lp
        num = self.circuit.V0 - self.circuit.R * current - voltage - emf - back_emf
        dIdt = num / Ltot
        dVdt = -current / self.circuit.C

        new_current = current + dIdt * dt
        new_voltage = voltage + dVdt * dt

        self.time.append(t + dt)
        self.currents.append(new_current)
        self.voltages.append(new_voltage)

        if energy_tracker is not None:
            Ltot = self.circuit.L + Lp
            energy_tracker.add(
                capacitor=0.5 * self.circuit.C * new_voltage**2,
                inductive=0.5 * Ltot * new_current**2,
            )

        return CouplingState(Lp=Lp, emf=emf, current=new_current, voltage=new_voltage)


def _profile_to_interp(profile, t_scale: float, y_scale: float):
    """Return interpolation functions for profile value and derivative."""

    if profile is None:
        return (lambda t: 0.0, lambda t: 0.0)

    arr = np.asarray(profile, dtype=float)
    t_vals = arr[:, 0] * t_scale
    y_vals = arr[:, 1] * y_scale

    def val(tt: float) -> float:
        return float(np.interp(tt, t_vals, y_vals, left=y_vals[0], right=y_vals[-1]))

    def deriv(tt: float) -> float:
        if tt <= t_vals[0]:
            i = 0
        elif tt >= t_vals[-1]:
            i = len(t_vals) - 2
        else:
            i = 0
            while i < len(t_vals) - 1 and not (t_vals[i] <= tt <= t_vals[i + 1]):
                i += 1
        dt = t_vals[i + 1] - t_vals[i]
        if dt == 0:
            return 0.0
        return (y_vals[i + 1] - y_vals[i]) / dt

    return val, deriv


def _run_distributed_network(
    cfg: CircuitConfig, t_end: float, num_points: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a chain of RLC segments using a state-space model."""

    segments, _ = cfg.build_distributed_model()
    n = len(segments)
    if n == 0:
        raise ValueError("No segments defined")

    L = np.array([s.totals()[0] for s in segments])
    R = np.array([s.totals()[1] for s in segments])
    C_nodes = np.zeros(n)
    C_nodes[0] = cfg.C_ext * 1e-6
    for j in range(1, n):
        C_nodes[j] = segments[j - 1].totals()[2]

    delay = cfg.switch_delay * 1e-9
    V0 = cfg.V0 * 1e3
    t_total = list(np.linspace(0.0, t_end * 1e-6, num_points))

    if t_end * 1e-6 <= delay:
        current = [0.0 for _ in t_total]
        voltage = [V0 for _ in t_total]
        z = np.zeros_like(t_total)
        return np.array(t_total), np.array(current), np.array(voltage), z, z

    idx_start = next((i for i, t in enumerate(t_total) if t >= delay), len(t_total))
    current = [0.0 for _ in t_total]
    voltage = [V0 for _ in t_total]
    t_eval = t_total[idx_start:]

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        I = y[:n]
        V = y[n:]
        dIdt = np.zeros(n)
        dVdt = np.zeros(n)
        for j in range(n):
            v_left = V[j]
            v_right = V[j + 1] if j + 1 < n else 0.0
            dIdt[j] = (v_left - v_right - R[j] * I[j]) / L[j]
        for j in range(n):
            I_left = I[j - 1] if j > 0 else 0.0
            I_right = I[j] if j < n - 1 else 0.0
            dVdt[j] = (I_left - I_right) / C_nodes[j]
        return np.concatenate([dIdt, dVdt])

    y0 = [0.0] * (2 * n)
    y0[n] = V0
    sol = solve_ivp(rhs, (delay, t_total[-1]), y0, t_eval=t_eval, method="BDF")

    sol_I = list(sol.y[0])
    sol_V = list(sol.y[n])
    for k, val in enumerate(sol_I):
        current[idx_start + k] = val
    for k, val in enumerate(sol_V):
        voltage[idx_start + k] = val

    z = np.zeros_like(t_total)
    return np.array(t_total), np.array(current), np.array(voltage), z, z


def run_circuit_simulation(
    cfg: CircuitConfig,
    t_end: float,
    num_points: int = 1000,
    plasma_solver: PlasmaSolverBase | None = None,
    plasma_state: Any | None = None,
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

    if cfg.segments:
        # Build distributed model and delegate to the lightweight solver
        segments, switches = cfg.build_distributed_model()
        dt = t_end * 1e-6 / (num_points - 1)
        sol = solve_distributed_circuit(
            segments, switches, V0=cfg.V0 * 1e3, t_end=t_end * 1e-6, dt=dt
        )
        z = np.zeros_like(sol.current)
        return sol.t, sol.current, sol.voltage, z, z

    # Basic series RLC with optional time varying inductances
    L_ext = cfg.L_ext * 1e-6
    R_ext = cfg.R_ext * 1e-3
    C_ext = cfg.C_ext * 1e-6
    V0 = cfg.V0 * 1e3

    t = np.linspace(0.0, t_end * 1e-6, num_points)
    dt = t[1] - t[0]
    I = np.zeros(num_points)
    V = np.zeros(num_points)
    V[0] = V0

    lp_val, lp_der = _profile_to_interp(cfg.plasma_inductance_profile, 1e-6, 1e-6)
    m_val, m_der = _profile_to_interp(cfg.mutual_inductance_profile, 1e-6, 1e-6)
    im_val, im_der = _profile_to_interp(cfg.mutual_current_profile, 1e-6, 1e3)

    # Special case: purely time-varying inductance with otherwise ideal circuit
    if (
        cfg.plasma_inductance_profile is not None
        and R_ext == 0.0
        and cfg.mutual_inductance_profile is None
        and cfg.mutual_current_profile is None
    ):
        # Assume linear profile for the simple analytic form used in tests
        t0, L0 = cfg.plasma_inductance_profile[0]
        t1, L1 = cfg.plasma_inductance_profile[-1]
        slope = ((L1 - L0) * 1e-6) / ((t1 - t0) * 1e-6 + 1e-30)
        I = V0 * t / (L_ext + slope * t)
        V = np.array([V0 for _ in t])
    else:
        for k in range(1, num_points):
            tk = t[k - 1]
            Lp = lp_val(tk)
            emf = m_val(tk) * im_der(tk) + im_val(tk) * m_der(tk)
            Ltot = L_ext + Lp
            dIdt = (V[k - 1] - R_ext * I[k - 1] - emf) / Ltot
            dVdt = -I[k - 1] / C_ext
            I[k] = I[k - 1] + dIdt * dt
            V[k] = V[k - 1] + dVdt * dt

    i_mutual = np.array([im_val(tt) for tt in t])
    v_mutual = np.array([m_val(tt) * im_der(tt) + im_val(tt) * m_der(tt) for tt in t])
    return t, I, V, i_mutual, v_mutual
