from __future__ import annotations

"""Lightweight RLC circuit with mutual inductance and back‑EMF coupling.

The solver advances a simple series RLC circuit while allowing optional
coupling to a plasma model through time‑dependent inductance and mutual
inductance terms.  A second‑order Runge–Kutta (mid‑point) update is used in
place of the original explicit Euler step for improved stability.  The update
enforces a Courant–Friedrichs–Lewy (CFL) condition of ``dt/sqrt(LC) < 0.5`` to
guard against unstable integrations.

The class exposes ``time``, ``currents`` and ``voltages`` attributes storing
history arrays that may be inspected after a simulation run.
"""

from dataclasses import dataclass, field
from typing import Callable

from .bases import CircuitSolverBase, CouplingState


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
    last_feedback: CouplingState = field(default_factory=CouplingState, init=False)

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
        coupling: CouplingState,
        back_emf: float,
        dt: float,
    ) -> CouplingState:
        """Advance the circuit state by ``dt`` seconds."""

        current = coupling.current
        voltage = coupling.voltage
        t = self.time[-1]

        Lp = coupling.Lp
        emf = coupling.emf

        # Time derivative of the plasma inductance feeds back on the circuit
        # through an induced voltage ``I * dLp/dt``.  Approximate ``dLp/dt`` using
        # the previous coupling state recorded on ``last_feedback``.
        dLp_dt = (Lp - self.last_feedback.Lp) / dt if dt > 0.0 else 0.0

        self.last_feedback = CouplingState(
            Lp=Lp, emf=emf, current=current, voltage=voltage
        )

        M, dIm_dt = self._mutual_terms(t, dt)

        Ltot = self.L_ext + Lp
        V_mutual = -M * dIm_dt

        # CFL condition based on LC time scale
        cfl = dt / (Ltot * self.C_ext) ** 0.5
        if cfl >= 0.5:
            raise ValueError(
                f"CFL condition violated (dt/sqrt(LC)={cfl:.3f} >= 0.5). Reduce dt."
            )

        def rhs(I: float, V: float) -> tuple[float, float]:
            numerator = (
                self.V0 + V_mutual - self.R_ext * I - V - emf - back_emf - I * dLp_dt
            )
            dIdt = numerator / Ltot
            dVdt = -I / self.C_ext
            return dIdt, dVdt

        # Second-order Runge--Kutta (mid-point) integration
        dI1, dV1 = rhs(current, voltage)
        I_mid = current + 0.5 * dI1 * dt
        V_mid = voltage + 0.5 * dV1 * dt
        dI2, dV2 = rhs(I_mid, V_mid)

        new_current = current + dI2 * dt
        new_voltage = voltage + dV2 * dt

        self.time.append(t + dt)
        self.currents.append(new_current)
        self.voltages.append(new_voltage)

        return CouplingState(
            Lp=Lp,
            emf=emf,
            current=new_current,
            voltage=new_voltage,
            mutual_inductance=M,
            back_reaction=V_mutual,
        )


__all__ = ["RLCCircuitSolver"]
