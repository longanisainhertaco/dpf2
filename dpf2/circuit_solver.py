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

__all__ = ["CircuitSolver", "RLCCircuit"]


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
