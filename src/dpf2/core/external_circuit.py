from __future__ import annotations

from dataclasses import dataclass

from .bases import CircuitSolverBase


@dataclass
class ExternalCircuit(CircuitSolverBase):
    """Minimal external circuit model.

    The circuit stores a time-varying inductance and integrates the current in
    response to an applied voltage using the simple relation ``V = L dI/dt``.
    This is intended solely for unit tests and does not represent a complete
    circuit model.
    """

    inductance: float = 1.0
    current: float = 0.0

    def step(
        self,
        current: float,
        voltage: float,
        dt: float,
        plasma_feedback: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Advance the circuit state by ``dt`` seconds.

        ``plasma_feedback`` may supply the instantaneous plasma inductance and
        its time derivative via the keys ``"Lp"`` and ``"dLpdt"`` respectively.
        """

        Lp = 0.0
        dLpdt = 0.0
        if plasma_feedback:
            Lp = plasma_feedback.get("Lp", 0.0)
            dLpdt = plasma_feedback.get("dLpdt", 0.0)

        Ltot = self.inductance + Lp
        dI = (voltage - dLpdt * current) * dt / max(Ltot, 1.0e-30)
        self.current = current + dI
        self.inductance = Ltot
        return self.current, voltage


__all__ = ["ExternalCircuit"]
