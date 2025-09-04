from __future__ import annotations

"""Spectral PSATD field solver with basic boundary handling.

The solver operates in one spatial dimension and is intended for unit tests
and lightweight examples.  It supports several idealised boundary conditions
and returns a divergence metric for diagnostics.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..physics.em_wave import EPS0

Boundary = Literal["periodic", "PEC", "PMC", "PML"]


@dataclass
class PSATDSolver:
    """Very small 1-D pseudo-spectral analytic time-domain solver.

    A lightweight implementation that avoids a full FFT dependency by using a
    direct integration of Gauss' law.  The interface mimics a spectral solver
    and provides a divergence metric for diagnostics.
    """

    num_cells: int
    length: float
    boundary: Boundary = "periodic"
    pml_cells: int = 0
    pml_sigma: float = 0.0

    def __post_init__(self) -> None:
        self.dx = self.length / self.num_cells if self.num_cells else 1.0

    # ------------------------------------------------------------------
    def solve(self, rho: np.ndarray) -> tuple[np.ndarray, float]:
        """Return electric field and divergence metric for given charge."""

        if getattr(rho, "size", len(rho)) != self.num_cells:
            raise ValueError("rho must have num_cells entries")

        # Cumulative integral of charge density
        total = 0.0
        values = []
        for val in rho:
            total += val
            values.append(total * self.dx / EPS0)
        E = np.array(values)
        E -= np.mean(E)

        # Apply simple boundary options
        if self.boundary == "PEC":
            E[0] = 0.0
            E[-1] = 0.0
        elif self.boundary == "PMC":
            E[0] = E[1]
            E[-1] = E[-2]
        elif self.boundary == "PML" and self.pml_cells:
            import math

            damp = [math.exp(-self.pml_sigma * i) for i in range(self.pml_cells, 0, -1)]
            for i, coef in enumerate(damp):
                E[i] *= coef
                E[-(i + 1)] *= coef

        dEdx = np.gradient(E, self.dx, edge_order=2)
        gauss = rho / EPS0
        divergence_error = float(abs(np.sum(dEdx - gauss)))
        return E, divergence_error


__all__ = ["PSATDSolver"]
