from __future__ import annotations

"""Simple multi-group radiation diffusion model.

The implementation intentionally targets a minimal subset of the full
multi-group radiation diffusion equations required for unit testing.  It
supports an arbitrary number of radiation energy groups and evolves their
energy densities via an implicit finite difference scheme.  A basic flux
limiter enforces the free-streaming limit at cell interfaces.  The model also
provides a coupling routine that exchanges energy between the fluid energy
equation and the radiation groups using user supplied opacities.
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

    def _limit_flux(self, F: float, left: float, right: float) -> float:
        """Limit a diffusive flux to the free-streaming value.

        The flux-limiter enforces ``|F| <= 0.5 * c * (E_L + E_R)`` where
        ``E_L`` and ``E_R`` are the radiation energy densities on either side
        of the interface.  This mimics a simple flux-limited diffusion model.

        Parameters
        ----------
        F:
            Proposed diffusive flux.
        left, right:
            Radiation energy densities on the left and right of the interface.

        Returns
        -------
        float
            The limited flux respecting the free-streaming bound.
        """

        limit = 0.5 * self.c * (left + right)
        if F > limit:
            return limit
        if F < -limit:
            return -limit
        return F

    def diffuse(self, dx: float, dt: float) -> List[List[float]]:
        """Advance the radiation energy densities by a diffusion step.

        A backward-Euler implicit discretisation with zero-flux boundaries is
        solved for each group using a tridiagonal Thomas algorithm.  After the
        implicit solve the diffusive flux at each interface is limited to the
        free-streaming value ``±c E`` to mimic flux-limited diffusion.  The
        method modifies ``self.energy`` in place and returns it for
        convenience.
        """

        E = self.energy
        groups = len(E)
        cells = len(E[0])
        coeffs = self.diffusion_coefficients
        for g in range(groups):
            D = coeffs[g]
            alpha = D * dt / (dx * dx)
            a = [0.0] * cells
            b = [0.0] * cells
            c = [0.0] * cells
            d = E[g][:]
            b[0] = 1.0 + alpha
            c[0] = -alpha
            for i in range(1, cells - 1):
                a[i] = -alpha
                b[i] = 1.0 + 2.0 * alpha
                c[i] = -alpha
            if cells > 1:
                a[cells - 1] = -alpha
                b[cells - 1] = 1.0 + alpha
            # Forward sweep
            for i in range(1, cells):
                m = a[i] / b[i - 1] if b[i - 1] != 0 else 0.0
                b[i] -= m * c[i - 1]
                d[i] -= m * d[i - 1]
            # Back substitution
            E_new = [0.0] * cells
            if cells:
                E_new[-1] = d[-1] / b[-1]
                for i in range(cells - 2, -1, -1):
                    denom = b[i] if b[i] != 0 else 1.0
                    E_new[i] = (d[i] - c[i] * E_new[i + 1]) / denom

            # Flux limiter
            flux = [0.0] * (cells + 1)
            for i in range(cells - 1):
                grad = (E_new[i + 1] - E_new[i]) / dx
                F = -D * grad
                F = self._limit_flux(F, E_new[i], E_new[i + 1])
                flux[i + 1] = F
            for i in range(cells):
                E_new[i] += dt / dx * (flux[i] - flux[i + 1])
            E[g] = E_new
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
