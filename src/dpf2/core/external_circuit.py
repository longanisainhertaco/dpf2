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

    def step(self, current: float, voltage: float, dt: float) -> tuple[float, float]:
        dI = voltage * dt / max(self.inductance, 1.0e-30)
        self.current = current + dI
        return self.current, voltage


__all__ = ["ExternalCircuit"]
