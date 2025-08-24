from __future__ import annotations

"""Equation of state (EOS) models used by the DPF solver.

This module provides a minimal interface for pressure and energy
calculations.  It now supports tabulated equations of state that supply
pressure and specific internal energy as functions of density and
temperature.  Interpolation is performed using SciPy's
``RegularGridInterpolator`` which yields bilinear behaviour on a regular
grid.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root_scalar

from ..core_schema import EOSModel
from dpf2.simulation.eos import TabulatedEOS as _SimulationTabulatedEOS

__all__ = ["EOSBase", "IdealGasEOS", "TabulatedEOS", "RealGasEOS", "create_eos"]


class EOSBase(Protocol):
    """Common EOS interface."""

    def pressure(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return pressure for density ``rho`` and temperature ``T``."""

    def energy(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return specific internal energy for density ``rho`` and temperature ``T``."""

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Return temperature for density ``rho`` and specific internal energy ``e``."""


@dataclass
class IdealGasEOS:
    """Ideal gas with constant heat capacity."""

    gamma: float = 5.0 / 3.0

    def pressure(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        return (self.gamma - 1.0) * rho * self.energy(rho, T)

    def energy(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:  # noqa: ARG002
        cv = 1.0 / (self.gamma - 1.0)
        return cv * T

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:  # noqa: ARG002
        cv = 1.0 / (self.gamma - 1.0)
        return e / cv


# ``RealGasEOS`` is kept for backwards compatibility with older tests.
RealGasEOS = IdealGasEOS


class TabulatedEOS:
    """Equation of state based on tabulated density/temperature data."""

    def __init__(
        self,
        filename: str | Path | dict[str, str | Path],
        mixture_fractions: dict[str, float] | str | None = None,
    ) -> None:
        """Load EOS tables from ``filename``.

        The underlying implementation reuses
        :class:`dpf2.simulation.eos.TabulatedEOS` to avoid code
        duplication.  The table must contain ``rho`` and ``T`` axes and
        datasets ``p`` (pressure) and ``e`` (specific internal energy).
        """

        self._impl = _SimulationTabulatedEOS(
            filename, mixture_fractions=mixture_fractions
        )
        # Convenience references used for interpolation and inversion
        self.rho_grid = self._impl.rho_grid
        self.T_grid = self._impl.T_grid
        self.p_interp: RegularGridInterpolator = self._impl.p_interp
        self.e_interp: RegularGridInterpolator = self._impl.e_interp

    def pressure(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Interpolate pressure for density ``rho`` and temperature ``T``."""

        points = np.stack([rho, T], axis=-1)
        return self.p_interp(points)

    def energy(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Interpolate energy for density ``rho`` and temperature ``T``."""

        points = np.stack([rho, T], axis=-1)
        return self.e_interp(points)

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Invert the energy table to obtain temperature."""

        def _solve(rho_val: float, e_val: float) -> float:
            def func(T):
                return self.energy(np.array([rho_val]), np.array([T]))[0] - e_val

            result = root_scalar(func, bracket=[self.T_grid[0], self.T_grid[-1]])
            return result.root

        return np.vectorize(_solve)(rho, e)


def create_eos(
    model: EOSModel, *, table_path: Path | None = None, gamma: float = 5.0 / 3.0
) -> EOSBase:
    """Factory for EOS implementations."""

    if model is EOSModel.IDEAL:
        return IdealGasEOS(gamma=gamma)
    if model is EOSModel.TABULATED and table_path is not None:
        return TabulatedEOS(Path(table_path))
    raise ValueError(f"Unsupported EOS model: {model}")

