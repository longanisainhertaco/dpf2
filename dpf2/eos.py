from __future__ import annotations

"""Thermodynamic helper models for DPF simulations."""

import numpy as np

__all__ = ["RealGasEOS"]


class RealGasEOS:
    """Ideal gas with temperature-dependent specific heat."""

    def __init__(self, gamma: float = 1.4, cp0: float = 1.0e4, cp_slope: float = 1.0):
        self.gamma = gamma
        self.cp0 = cp0
        self.cp_slope = cp_slope

    # ------------------------------------------------------------------
    def cp(self, T: float | np.ndarray) -> float | np.ndarray:
        """Specific heat capacity C_p(T) in J/(kg K)."""
        return self.cp0 + self.cp_slope * T

    def cv(self, T: float | np.ndarray) -> float | np.ndarray:
        return self.cp(T) / self.gamma

    def pressure(self, density: float | np.ndarray, T: float | np.ndarray) -> float | np.ndarray:
        """Ideal gas equation of state P = rho * R_specific * T."""
        R_specific = (self.gamma - 1.0) * self.cp(T)
        return density * R_specific * T

