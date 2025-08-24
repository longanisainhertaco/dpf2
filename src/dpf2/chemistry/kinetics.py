from __future__ import annotations

"""Time dependent ionisation and recombination kinetics.

This module provides a tiny collisional\u2013radiative model that evolves the
free electron density using tabulated ionisation and recombination rate
coefficients.  The implementation is intentionally simple and intended for
unit testing of chemistry hooks rather than high fidelity simulation.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RateTable:
    """Tabulated ionisation and recombination rate coefficients."""

    T: np.ndarray
    k_ion: np.ndarray
    k_rec: np.ndarray

    @classmethod
    def from_csv(cls, path: str | Path) -> "RateTable":
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        return cls(T=data[:, 0], k_ion=data[:, 1], k_rec=data[:, 2])

    def ion_rate(self, T: np.ndarray) -> np.ndarray:
        return np.interp(T, self.T, self.k_ion, left=self.k_ion[0], right=self.k_ion[-1])

    def rec_rate(self, T: np.ndarray) -> np.ndarray:
        return np.interp(T, self.T, self.k_rec, left=self.k_rec[0], right=self.k_rec[-1])


@dataclass
class RateEquations:
    """Time dependent ionisation model."""

    rates: RateTable

    def rhs(self, ne: float, n_total: float, T: float) -> float:
        """Time derivative of the electron density."""

        k_i = float(self.rates.ion_rate(np.asarray([T])))
        k_r = float(self.rates.rec_rate(np.asarray([T])))
        ion = k_i * ne * (n_total - ne)
        rec = k_r * ne * ne
        return ion - rec

    def step(self, ne: float, n_total: float, T: float, dt: float) -> float:
        """Advance ``ne`` by a single explicit Euler step."""

        return ne + dt * self.rhs(ne, n_total, T)


__all__ = ["RateTable", "RateEquations"]
