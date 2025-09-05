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
from ..diagnostics.quality_dashboard import QualityDashboard


EPS0 = 1.0  # Permittivity used for the lightweight solvers


def lhdi_resistivity(
    n: np.ndarray, B: np.ndarray, dx: float, coeff: float = 1.0
) -> np.ndarray:
    """Return a simple lower-hybrid drift resistivity estimate.

    The model mirrors the implementation used in the full PIC solver by
    taking the magnitude of the density and magnetic-field gradients and
    returning ``coeff * |∇n| * |∇B|``.  The helper is intentionally lightweight
    so that unit tests can exercise LHDI-driven resistivity without pulling in
    heavier solver components.
    """

    if len(n) == 0 or len(B) == 0:
        return np.zeros_like(n)
    try:
        grad_n = np.gradient(n, dx)
        grad_B = np.gradient(B, dx)
        eta = coeff * np.abs(grad_n) * np.abs(grad_B)
    except Exception:  # pragma: no cover - minimal numpy stubs
        eta = np.zeros_like(n)
    return eta


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
    quality: QualityDashboard | None = None
    circuit_feedback: CouplingState = field(init=False, default_factory=CouplingState)
    divergence_error: float = field(init=False, default=0.0)
    energy_drift: float = field(init=False, default=0.0)
    wave_power: float = field(init=False, default=0.0)
    wave_spectrum: np.ndarray | None = field(init=False, default=None)
    axial_field: float = field(init=False, default=0.0)
    last_eta: np.ndarray | None = field(init=False, default=None)
    _prev_energy: float = field(init=False, default=0.0)
    _step_count: int = field(init=False, default=0)

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

            try:
                spectrum = np.abs(np.fft.rfft(self.E))
            except Exception:  # pragma: no cover - ``numpy`` stub fallback
                spectrum = np.zeros(0)
            self.wave_spectrum = spectrum
            self.wave_power = float(np.sum(spectrum ** 2)) if len(spectrum) else 0.0
            self.axial_field = float(np.mean(self.E))
            self.last_eta = lhdi_resistivity(np.abs(self.rho), np.abs(self.E), self.dx)
        else:
            self.wave_spectrum = None
            self.wave_power = 0.0
            self.axial_field = voltage / self.length if self.length else 0.0
            self.last_eta = None

        if self.quality is not None:
            self._step_count += 1
            cell_size = self.dx if self.field_solver != "circuit" else self.length
            total_particles = len(self.positions)
            ppc = total_particles / self.num_cells if self.num_cells else 0.0
            max_v = max(abs(v) for v in self.velocities) if self.velocities else 0.0
            cfl = max_v * dt / cell_size if cell_size else 0.0
            lambda_D = cell_size  # placeholder for Debye length estimate
            plasma_impedance = (
                float(np.mean(self.last_eta)) if self.last_eta is not None else 0.0
            )
            self.quality.log(
                self._step_count,
                dt,
                cell_size,
                ppc,
                cfl,
                lambda_D,
                lower_hybrid_power=self.wave_power,
                plasma_impedance=plasma_impedance,
                divergence_error=self.divergence_error,
                energy_drift=self.energy_drift,
            )

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


__all__ = ["SimplePIC", "HybridPIC", "lhdi_resistivity"]
