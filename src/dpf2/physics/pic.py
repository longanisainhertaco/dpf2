from __future__ import annotations

"""Lightweight particle-in-cell (PIC) solver.

The solver is intentionally compact and targets unit tests and educational
examples.  It advances a collection of macro particles in one spatial
dimension under the influence of the electric field inferred from the circuit
voltage.  A simple current estimate provides a coupling back to the circuit via
:class:`~dpf2.core.bases.CouplingState`.
"""

from dataclasses import dataclass, field
from typing import Any, List

from ..core.bases import PlasmaSolverBase, CouplingState


@dataclass
class SimplePIC(PlasmaSolverBase):
    """Very small 1‑D PIC model used for tests and examples.

    Parameters
    ----------
    charge, mass:
        Particle charge and mass in SI units.  All particles are assumed to be
        identical macro particles.
    length:
        Length of the one‑dimensional domain.  Periodic boundaries are
        imposed; particles leaving the domain wrap around to the opposite end.
    positions, velocities:
        Initial particle phase‑space coordinates.  The arrays are modified in
        place by :meth:`step`.
    """

    charge: float
    mass: float
    length: float
    positions: List[float]
    velocities: List[float]
    circuit_feedback: CouplingState = field(init=False, default_factory=CouplingState)

    def step(self, state: Any, dt: float, current: float, voltage: float) -> Any:
        """Advance particles by ``dt`` and compute circuit feedback."""

        # Electric field assumed uniform over the domain.
        E = voltage / self.length if self.length else 0.0
        accel = self.charge / self.mass * E
        self.velocities = [v + accel * dt for v in self.velocities]
        self.positions = [
            (x + v * dt) % self.length if self.length else x + v * dt
            for x, v in zip(self.positions, self.velocities)
        ]
        # Estimate the plasma current from particle motion.
        plasma_current = (
            self.charge * sum(self.velocities) / self.length if self.length else 0.0
        )
        self.circuit_feedback = CouplingState(
            current=current, voltage=voltage, back_reaction=plasma_current
        )
        return state

    def coupling_interface(self) -> CouplingState:  # pragma: no cover - simple
        """Expose the latest coupling information."""

        return CouplingState(back_reaction=self.circuit_feedback.back_reaction)


__all__ = ["SimplePIC"]
