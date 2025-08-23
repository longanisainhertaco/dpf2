"""Skeleton Hall-MHD solver with placeholders for advanced physics.

This module defines data structures and a minimal solver interface for a
future 3-D resistive Hall-MHD implementation with anisotropic transport,
constrained transport (CT) for divergence-free magnetic fields, and
hooks for AMR integration.  The solver is intentionally incomplete but
provides typed containers and method stubs so that further development
can proceed incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import PlasmaSolverBase

__all__ = ["MHDState", "HallMHDSolver"]


@dataclass
class MHDState:
    """State container for the MHD variables.

    Attributes
    ----------
    rho : ndarray
        Mass density [kg/m^3].
    mom : ndarray
        Momentum density vector [kg/m^2/s].
    energy : ndarray
        Total energy density [J/m^3].
    B : ndarray
        Magnetic field vector [T].
    Te : ndarray | None
        Electron temperature [K] when using two-temperature models.
    Ti : ndarray | None
        Ion temperature [K] when using two-temperature models.
    """

    rho: np.ndarray
    mom: np.ndarray
    energy: np.ndarray
    B: np.ndarray
    Te: np.ndarray | None = None
    Ti: np.ndarray | None = None


@dataclass
class HallMHDSolver(PlasmaSolverBase):
    """Stub for a 3-D Hall-MHD solver with CT and AMR hooks.

    Parameters
    ----------
    mesh : Any
        Placeholder for mesh/AMR hierarchy object.
    """

    mesh: Any = field(default=None)

    def step(self, state: MHDState, dt: float) -> MHDState:  # pragma: no cover - skeleton
        """Advance the state by ``dt`` seconds.

        This placeholder method documents the intended numerical
        operations:

        1. Compute fluxes using an HLLD Riemann solver.
        2. Update conserved variables with a high-order Godunov scheme.
        3. Apply constrained transport to maintain ∇·B = 0.
        4. Add Hall, Nernst and resistive source terms (potentially
           via IMEX).
        5. Couple radiation and EOS modules through source terms.

        The method currently returns the input state unchanged.
        """

        # TODO: Implement full solver.
        return state
