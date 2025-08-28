from __future__ import annotations

"""Finite-difference time-domain (FDTD) solver for 1-D Maxwell equations.

The implementation is intentionally compact and targets unit tests.  It supports
coupling to the distributed circuit solver via the :class:`PlasmaSolverBase`
interface.  Circuit voltages and currents are interpreted as the tangential
electric and magnetic field at the left boundary of the domain.  The resulting
boundary current is fed back to the circuit through the
:class:`~dpf2.core.bases.CouplingState` ``back_reaction`` term.

The solver evolves the fields on an unstructured (potentially non-uniform)
1-D mesh defined by the cell lengths supplied at construction time.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence, List
import math

from ..core.bases import PlasmaSolverBase, CouplingState

# Physical constants in SI units
EPS0 = 8.8541878128e-12
MU0 = 4.0 * math.pi * 1e-7
C0 = 1.0 / math.sqrt(EPS0 * MU0)


@dataclass
class FDTDSolver(PlasmaSolverBase):
    """Very small 1-D FDTD Maxwell solver.

    Parameters
    ----------
    lengths:
        Sequence of cell lengths defining the spatial discretisation.  The
        electric field ``E`` is stored at cell vertices and the magnetic field
        ``H`` at cell centres.
    eps, mu:
        Permittivity and permeability of the medium.  Defaults to vacuum
        values.
    """

    lengths: Sequence[float]
    eps: float = EPS0
    mu: float = MU0
    E: List[float] = field(init=False)
    H: List[float] = field(init=False)
    circuit_feedback: CouplingState = field(init=False, default_factory=CouplingState)

    def __post_init__(self) -> None:  # pragma: no cover - trivial initialisation
        self.dx = list(self.lengths)
        n = len(self.dx)
        # E at vertices -> n+1 entries, H at cell centres -> n entries
        self.E = [0.0] * (n + 1)
        self.H = [0.0] * n

    # ------------------------------------------------------------------
    def step(self, state: Any, dt: float, current: float, voltage: float) -> Any:
        """Advance fields by ``dt`` using a Yee-style scheme."""

        if not self.dx:
            return state

        # Left boundary is driven by the circuit
        self.E[0] = voltage / self.dx[0] if self.dx[0] else 0.0
        self.H[0] = current

        # Update H field (curl E)
        for i in range(len(self.H)):
            dE = self.E[i + 1] - self.E[i]
            self.H[i] = self.H[i] + dt / (self.mu * self.dx[i]) * dE

        # Update E field (curl H)
        for i in range(1, len(self.E) - 1):
            dH = self.H[i] - self.H[i - 1]
            self.E[i] = self.E[i] + dt / (self.eps * self.dx[i - 1]) * dH

        # Simple absorbing boundary at the right end
        self.E[-1] = 0.0

        boundary_current = self.H[0]
        self.circuit_feedback = CouplingState(
            current=current, voltage=voltage, back_reaction=boundary_current
        )
        return state

    # ------------------------------------------------------------------
    def coupling_interface(self) -> CouplingState:  # pragma: no cover - simple
        return CouplingState(back_reaction=self.circuit_feedback.back_reaction)

    # ------------------------------------------------------------------
    def field_at(self, x: float) -> float:
        """Return the electric field at position ``x`` along the domain."""
        if not self.dx:
            return 0.0
        pos = 0.0
        for i, d in enumerate(self.dx):
            if x < pos + d:
                # Linear interpolation within the cell
                t = (x - pos) / d if d else 0.0
                return (1 - t) * self.E[i] + t * self.E[i + 1]
            pos += d
        return self.E[-1]


__all__ = ["FDTDSolver", "EPS0", "MU0", "C0"]
