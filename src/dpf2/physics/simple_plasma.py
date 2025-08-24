from __future__ import annotations

"""Toy plasma solver providing inductance feedback to the circuit model."""

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.bases import PlasmaSolverBase


@dataclass
class ZeroDPlasma(PlasmaSolverBase):
    """Very small plasma model used for tests and examples.

    The solver keeps a dummy state and uses a user supplied ``inductance``
    function to provide coupling terms back to the circuit.  The function is
    expected to return ``(Lp, dLpdt)`` when called with the current simulation
    time, current and voltage.
    """

    inductance: Callable[[float, float, float], tuple[float, float]]
    time: float = 0.0
    circuit_feedback: dict[str, float] = field(init=False, default_factory=dict)

    def step(self, state: Any, dt: float, current: float, voltage: float) -> Any:
        """Advance the dummy plasma state and compute circuit feedback."""

        self.time += dt
        Lp, dLpdt = self.inductance(self.time, current, voltage)
        self.circuit_feedback = {"Lp": Lp, "dLpdt": dLpdt}
        return state


__all__ = ["ZeroDPlasma"]
