"""Ideal gas equation of state backend.

This lightweight implementation provides ion and electron pressure and
internal energy for a fully ionised or partially ionised ideal gas.  The
model assumes a constant adiabatic index ``gamma`` and mean molecular
weight ``mu`` (in arbitrary units).  When ``ionization`` is non-zero the
same amount of pressure and energy is attributed to the electron fluid.

The calculations intentionally avoid physical constants in order to keep
unit tests self contained.  The specific gas constant is taken as
``1 / mu`` so that pressure is given by ``p = rho * R * T`` and the
specific internal energy follows from ``e = R * T / (gamma - 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


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
        """Return the specific gas constant used by the model."""

        return 1.0 / self.mu

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

    def electron_energy(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Electron specific internal energy for density ``rho`` and temperature ``T``."""

        if self.ionization == 0.0:
            return np.zeros_like(rho)
        cv = self.R / (self.gamma - 1.0)
        return self.ionization * cv * T
