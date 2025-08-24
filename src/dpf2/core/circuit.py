from __future__ import annotations

"""Lightweight RLC circuit with mutual inductance and back‑EMF coupling.

The solver advances a simple series RLC circuit while allowing optional
coupling to a plasma model through time‑dependent inductance and mutual
inductance terms.  A minimal explicit Euler update is employed which is
sufficient for regression tests and examples.

The class exposes ``time``, ``currents`` and ``voltages`` attributes storing
history arrays that may be inspected after a simulation run.
"""

from dataclasses import dataclass, field
from typing import Callable

from .bases import CircuitSolverBase


@dataclass
class RLCCircuitSolver(CircuitSolverBase):
    """Series RLC circuit with optional plasma coupling.

    Parameters
    ----------
    L_ext, R_ext, C_ext, V0:
        External inductance, resistance, capacitance and initial capacitor
        voltage respectively (SI units).
    mutual_inductance:
        Optional callable ``M(t)`` returning the mutual inductance in Henries.
    mutual_current:
        Optional callable ``I_m(t)`` returning the current in the coupled
        circuit.  The time derivative is approximated by a finite difference.
    """

    L_ext: float
    R_ext: float
    C_ext: float
    V0: float
    mutual_inductance: Callable[[float], float] | None = None
    mutual_current: Callable[[float], float] | None = None

    time: list[float] = field(default_factory=lambda: [0.0])
    currents: list[float] = field(default_factory=lambda: [0.0])
    voltages: list[float] = field(init=False)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self.voltages = [self.V0]

    # ------------------------------------------------------------------
    def _mutual_terms(self, t: float, dt: float) -> tuple[float, float]:
        """Return ``(M, dI_m/dt)`` at time ``t`` using finite differences."""

        if self.mutual_inductance is None or self.mutual_current is None:
            return 0.0, 0.0
        M = self.mutual_inductance(t)
        Im_now = self.mutual_current(t)
        Im_next = self.mutual_current(t + dt)
        dIm_dt = (Im_next - Im_now) / dt
        return M, dIm_dt

    # ------------------------------------------------------------------
    def step(
        self,
        current: float,
        voltage: float,
        dt: float,
        plasma_feedback: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Advance the circuit state by ``dt`` seconds.

        Parameters
        ----------
        current, voltage:
            Present values of the circuit current and capacitor voltage.
        dt:
            Time step in seconds.
        plasma_feedback:
            Optional mapping containing coupling terms from the plasma solver.
            Recognised keys are ``"Lp"`` for plasma inductance (Henries) and
            either ``"dLpdt"`` for its time derivative or ``"emf"`` for the
            induced back‑EMF (Volts).  Additional keys ``"M"`` and
            ``"dIm_dt"`` may be supplied to override the mutual inductance
            values returned by the callables passed at construction time.
        """

        t = self.time[-1]
        Lp = 0.0
        dLpdt = 0.0
        emf = 0.0
        use_emf = False
        M_pf = None
        dIm_dt_pf = None
        if plasma_feedback:
            Lp = plasma_feedback.get("Lp", 0.0)
            if "emf" in plasma_feedback:
                emf = plasma_feedback["emf"]
                use_emf = True
            else:
                dLpdt = plasma_feedback.get("dLpdt", 0.0)
            M_pf = plasma_feedback.get("M")
            dIm_dt_pf = plasma_feedback.get("dIm_dt")

        M, dIm_dt = self._mutual_terms(t, dt)
        if M_pf is not None:
            M = M_pf
        if dIm_dt_pf is not None:
            dIm_dt = dIm_dt_pf

        Ltot = self.L_ext + Lp
        V_mutual = -M * dIm_dt
        if use_emf:
            dIdt = (self.V0 + V_mutual - self.R_ext * current - voltage - emf) / Ltot
        else:
            dIdt = (self.V0 + V_mutual - self.R_ext * current - voltage - dLpdt * current) / Ltot
        dVdt = -current / self.C_ext

        new_current = current + dIdt * dt
        new_voltage = voltage + dVdt * dt

        self.time.append(t + dt)
        self.currents.append(new_current)
        self.voltages.append(new_voltage)

        return new_current, new_voltage


__all__ = ["RLCCircuitSolver"]
