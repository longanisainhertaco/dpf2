"""Ideal gas equation of state backend.

This lightweight implementation provides ion and electron pressure and
specific internal energy for a fully or partially ionised ideal gas.  The
model assumes a constant adiabatic index ``gamma`` and mean molecular weight
``mu`` given in g/mol.  When ``ionization`` is non‑zero the same amount of
pressure and energy is attributed to the electron fluid.

The specific gas constant is derived from the universal gas constant ``R``
so that

``R_specific = R * 1000 / mu``

yielding pressure ``p = rho * R_specific * T`` and specific internal energy
``e = R_specific * T / (gamma - 1)``.  Densities are expected in kg/m^3,
temperatures in Kelvin, pressures in Pascal and energies in J/kg.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:  # pragma: no cover - prefer SciPy when available
    from scipy.constants import R as R_UNIVERSAL
except Exception:  # pragma: no cover - lightweight fallback
    R_UNIVERSAL = 8.31446261815324  # J/(mol*K)

GRAMS_PER_KILOGRAM = 1000.0


@dataclass
class IdealGasEOS:
    """Ideal‑gas equation of state with optional electron component."""

    gamma: float = 5.0 / 3.0
    mu: float = 1.0
    ionization: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def R(self) -> float:
        """Return the specific gas constant [J/(kg*K)]."""

        return R_UNIVERSAL * GRAMS_PER_KILOGRAM / self.mu

    # ------------------------------------------------------------------
    # EOS interface
    # ------------------------------------------------------------------
    def ion_pressure(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Ion pressure for density ``rho`` and temperature ``T``."""

        return rho * self.R * T

    def electron_pressure(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Electron pressure contribution for density ``rho`` and temperature ``T``."""

        if self.ionization == 0.0:
            return np.zeros_like(rho)
        return self.ionization * rho * self.R * T

    def ion_energy(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Ion specific internal energy for density ``rho`` and temperature ``T``."""

        cv = self.R / (self.gamma - 1.0)
        return cv * T

    def electron_energy(
        self, rho: np.ndarray, T: np.ndarray
    ) -> np.ndarray:  # noqa: ARG002
        """Electron specific internal energy for density ``rho`` and temperature ``T``."""

        if self.ionization == 0.0:
            return np.zeros_like(rho)
        cv = self.R / (self.gamma - 1.0)
        return self.ionization * cv * T
