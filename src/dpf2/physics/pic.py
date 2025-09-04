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

import math
import numpy as np

from ..core.bases import PlasmaSolverBase, CouplingState
from ..fields.psatd_solver import PSATDSolver


EPS0 = 1.0  # Permittivity used for the lightweight solvers


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
    field_solver: str = "circuit"
    deposition: str = "standard"
    num_cells: int = 64
    circuit_feedback: CouplingState = field(init=False, default_factory=CouplingState)
    divergence_error: float = field(init=False, default=0.0)
    energy_drift: float = field(init=False, default=0.0)
    _prev_energy: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.field_solver != "circuit":
            self.num_cells = max(int(self.num_cells), 1)
            self.dx = self.length / self.num_cells if self.length else 1.0
            self.E = np.zeros(self.num_cells)
            self.rho = np.zeros(self.num_cells)
            if self.field_solver == "PSATD":
                self._psatd_solver = PSATDSolver(self.num_cells, self.length)
        else:
            self.dx = self.length
            self.E = np.zeros(0)
            self.rho = np.zeros(0)
        if self.deposition not in {"standard", "Esirkepov", "EZ"}:
            raise ValueError("Unknown deposition scheme")

    # ------------------------------------------------------------------
    def _deposit(self) -> None:

        if len(self.rho) == 0:
            return

        for i in range(len(self.rho)):
            self.rho[i] = 0.0
        cell_vol = self.length / self.num_cells if self.length else 1.0
        import math

        for x in self.positions:
            if self.deposition == "EZ":
                cell = x / self.dx
                left = int(math.floor(cell))
                frac = cell - left
                self.rho[left % self.num_cells] += (
                    self.charge * (1.0 - frac) / cell_vol
                )
                self.rho[(left + 1) % self.num_cells] += (
                    self.charge * frac / cell_vol
                )
            else:  # standard and Esirkepov use simple nearest grid
                idx = int(math.floor(x / self.length * self.num_cells)) % self.num_cells
                self.rho[idx] += self.charge / cell_vol

    def _solve_fields(self) -> None:
        if self.field_solver == "PSATD":
            self.E, self.divergence_error = self._psatd_solver.solve(self.rho)
        else:
            self.E = np.cumsum(self.rho) * self.dx / EPS0
            self.E -= np.mean(self.E)
            dEdx = np.gradient(self.E, self.dx, edge_order=2)
            gauss = self.rho / EPS0
            self.divergence_error = float(abs(np.sum(dEdx - gauss)))


    def _interp_E(self, x: float) -> float:
        if len(self.E) == 0:
            return 0.0

        import math

        idx = int(math.floor(x / self.length * self.num_cells)) % self.num_cells

        return float(self.E[idx])

    def step(self, state: Any, dt: float, current: float, voltage: float) -> Any:
        """Advance particles by ``dt`` and compute circuit feedback."""

        if self.field_solver == "circuit":
            E = voltage / self.length if self.length else 0.0
            accel = self.charge / self.mass * E
            self.velocities = [v + accel * dt for v in self.velocities]
            self.positions = [
                (x + v * dt) % self.length if self.length else x + v * dt
                for x, v in zip(self.positions, self.velocities)
            ]
            plasma_current = (
                self.charge * sum(self.velocities) / self.length if self.length else 0.0
            )
            self.circuit_feedback = CouplingState(
                current=current, voltage=voltage, back_reaction=plasma_current
            )
            return state

        # Spectral/finite-difference solver
        self._deposit()
        self._solve_fields()
        accel_coeff = self.charge / self.mass
        self.velocities = [
            v + accel_coeff * self._interp_E(x) * dt
            for x, v in zip(self.positions, self.velocities)
        ]
        self.positions = [
            (x + v * dt) % self.length if self.length else x + v * dt
            for x, v in zip(self.positions, self.velocities)
        ]
        plasma_current = (
            self.charge * sum(self.velocities) / self.length if self.length else 0.0
        )
        self.circuit_feedback = CouplingState(
            current=current, voltage=voltage, back_reaction=plasma_current
        )

        # Diagnostics ------------------------------------------------------
        if len(self.E):

            field_energy = 0.5 * EPS0 * np.sum(self.E ** 2) * self.dx

            kinetic_energy = 0.5 * self.mass * sum(v**2 for v in self.velocities)
            total = field_energy + kinetic_energy
            self.energy_drift = (
                (total - self._prev_energy) / self._prev_energy
                if self._prev_energy
                else 0.0
            )
            self._prev_energy = total
        return state

    def coupling_interface(self) -> CouplingState:  # pragma: no cover - simple
        """Expose the latest coupling information."""

        return CouplingState(back_reaction=self.circuit_feedback.back_reaction)


@dataclass
class HybridPIC(SimplePIC):
    """Hybrid kinetic solver blending PIC particles with a fluid response.

    The model extends :class:`SimplePIC` by adding a prescribed fraction of the
    circuit current as a fluid contribution to the back‑reaction.  This keeps
    the implementation intentionally lightweight while exercising the coupling
    hooks in unit tests.
    """

    fluid_fraction: float = 0.0

    def step(self, state: Any, dt: float, current: float, voltage: float) -> Any:
        state = super().step(state, dt, current, voltage)
        fluid_current = current * self.fluid_fraction
        total = self.circuit_feedback.back_reaction + fluid_current
        self.circuit_feedback = CouplingState(
            current=current, voltage=voltage, back_reaction=total
        )
        return state


__all__ = ["SimplePIC", "HybridPIC"]
