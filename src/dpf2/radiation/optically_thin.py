"""Optically thin radiation closures with chemistry-aware emissivities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover
    from scipy.constants import m_p
except Exception:  # pragma: no cover
    m_p = 1.67262192369e-27

from ..chemistry import ChemistryModel
from .power import bremsstrahlung_power, line_radiation_power


@dataclass
class OpticallyThinRadiation:
    """Combine bremsstrahlung and line losses using a chemistry model."""

    chemistry: ChemistryModel
    line_coeff: float = 0.0
    impurity_fraction: float = 1.0
    Z_floor: float = 1.0

    def loss(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        zbar = np.maximum(self.chemistry.ionization_state(rho, T), self.Z_floor)
        ne = rho * zbar / m_p
        ni = rho / m_p
        brem = bremsstrahlung_power(ne, ni, T, Z_eff=zbar)
        lines = line_radiation_power(ne, T, coeff=self.line_coeff, impurity_fraction=self.impurity_fraction)
        return brem + lines


@dataclass
class GrayRadiationInterface:
    """Placeholder gray FLD/M1 interface used to plumb future transport."""

    radiation: OpticallyThinRadiation

    def source(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        return -self.radiation.loss(rho, T)

    def couple(self, energy: list[float], dt: float) -> list[float]:
        if not energy:
            return energy
        source = float(np.mean(self.radiation.loss(np.array([1.0]), np.array([1.0]))))
        return [e + dt * source for e in energy]


__all__ = ["OpticallyThinRadiation", "GrayRadiationInterface"]
