from __future__ import annotations

"""Simple multi-group radiation diffusion model.

The implementation intentionally targets a minimal subset of the full
multi-group radiation diffusion equations required for unit testing.  It
supports an arbitrary number of radiation energy groups and evolves their
energy densities via a first-order explicit finite difference scheme.  The
model also provides a coupling routine that exchanges energy between the
fluid energy equation and the radiation groups using user supplied opacities.
"""

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np

__all__ = ["MultiGroupDiffusion"]


@dataclass
class MultiGroupDiffusion:
    """Diffusion model for user-defined radiation energy groups.

    Parameters
    ----------
    opacities:
        Sequence of group opacities :math:`\kappa_g` [1/m].
    c:
        Speed of light used to compute diffusion coefficients.  A reduced
        value may be supplied for test problems.
    """

    opacities: List[float]
    c: float = 2.99792458e8

    def __post_init__(self) -> None:
        if any(o <= 0 for o in self.opacities):
            raise ValueError("opacities must be positive")
        self.group_count = len(self.opacities)
        # Energy density for each group and cell.  The array is initialised
        # with a single cell; it is resized on demand by :meth:`couple` if the
        # caller supplies more cells.
        self.energy = [[0.0] for _ in range(self.group_count)]

    # ------------------------------------------------------------------
    # Diffusion coefficients and step
    # ------------------------------------------------------------------
    @property
    def diffusion_coefficients(self) -> np.ndarray:
        """Return diffusion coefficient for each group.

        The standard approximation :math:`D_g = c/(3\kappa_g)` is used."""

        return np.array([self.c / (3.0 * kappa) for kappa in self.opacities])

    def diffuse(self, dx: float, dt: float) -> List[List[float]]:
        """Advance the radiation energy densities by a diffusion step.

        A simple explicit finite-difference update with zero-flux boundary
        conditions is employed.  The method modifies ``self.energy`` in place
        and returns it for convenience.
        """

        E = self.energy
        groups = len(E)
        cells = len(E[0])
        coeffs = self.diffusion_coefficients
        for g in range(groups):
            D = coeffs[g]
            # Compute fluxes with zero-flux boundaries
            flux = [0.0] * (cells + 1)
            for i in range(cells - 1):
                grad = (E[g][i + 1] - E[g][i]) / dx
                flux[i + 1] = -D * grad
            for i in range(cells):
                E[g][i] += dt / dx * (flux[i] - flux[i + 1])
        return E

    # ------------------------------------------------------------------
    # Coupling to fluid energy equation
    # ------------------------------------------------------------------
    def couple(
        self, fluid_energy: Union[Sequence[float], float], dt: float
    ) -> Union[List[float], float]:
        """Exchange energy with the fluid energy equation.

        Parameters
        ----------
        fluid_energy:
            Array of fluid energies per cell.  The method subtracts the energy
            transferred to the radiation groups from this array and returns the
            updated values.
        dt:
            Time step used for the coupling.
        """

        if isinstance(fluid_energy, (int, float)):
            fe = [float(fluid_energy)]
        else:
            fe = [float(x) for x in fluid_energy]
        cells = len(fe)
        if len(self.energy[0]) != cells:
            self.energy = [[0.0 for _ in range(cells)] for _ in range(self.group_count)]
        for i in range(cells):
            total_loss = 0.0
            for g in range(self.group_count):
                loss = self.opacities[g] * fe[i] * dt
                self.energy[g][i] += loss
                total_loss += loss
            fe[i] -= total_loss
        return fe if not isinstance(fluid_energy, (int, float)) else fe[0]
