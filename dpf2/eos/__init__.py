from __future__ import annotations

"""Equation of state (EOS) models used by the DPF solver.

This module provides a small interface for pressure and temperature
calculations used throughout the code base.  Only simplified
implementations are supplied – the goal is to expose an API that can be
extended with real SESAME/FPEOS table readers in the future.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..core_schema import EOSModel

__all__ = [
    "EOSBase",
    "IdealGasEOS",
    "TabulatedEOS",
    "RealGasEOS",
    "create_eos",
]


class EOSBase(Protocol):
    """Common EOS interface."""

    def pressure(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Return pressure for density ``rho`` and specific internal energy ``e``."""

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Return temperature for density ``rho`` and specific internal energy ``e``."""


@dataclass
class IdealGasEOS:
    """Ideal gas with constant heat capacity."""

    gamma: float = 5.0 / 3.0

    def pressure(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        return (self.gamma - 1.0) * rho * e

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:  # noqa: ARG002
        cv = 1.0 / (self.gamma - 1.0)
        return e / cv


# ``RealGasEOS`` is kept for backwards compatibility with older tests.
RealGasEOS = IdealGasEOS


class TabulatedEOS:
    """Equation of state based on a tabulated 2-D data set.

    The table is expected to be stored in an HDF5 file with datasets
    ``rho`` and ``e`` defining the grid axes and ``p`` and ``T`` providing
    the pressure and temperature values on that grid.  Bilinear
    interpolation is performed using :class:`scipy.interpolate.RegularGridInterpolator`.
    """

    def __init__(
        self,
        filename: str | Path | dict[str, str | Path],
        mixture_fractions: dict[str, float] | None = None,
    ):
        if mixture_fractions:
            if isinstance(filename, (str, Path)):
                base = Path(filename)
                species_files = {sp: base / f"{sp}.h5" for sp in mixture_fractions}
            elif isinstance(filename, dict):
                species_files = {sp: Path(path) for sp, path in filename.items()}
            else:
                raise TypeError(
                    "filename must be a path or mapping when mixture_fractions are provided"
                )

            first = True
            for species, path in species_files.items():
                with h5py.File(path, "r") as f:
                    if not all(key in f for key in ("rho", "e", "p", "T")):
                        raise ValueError("EOS table is missing required datasets.")
                    rho_grid = f["rho"][:]
                    e_grid = f["e"][:]
                    p_table = f["p"][:]
                    T_table = f["T"][:]

                weight = mixture_fractions.get(species, 0.0)
                if first:
                    self.rho_grid = rho_grid
                    self.e_grid = e_grid
                    self.p_table = weight * p_table
                    self.T_table = weight * T_table
                    first = False
                else:
                    if not (
                        np.array_equal(self.rho_grid, rho_grid)
                        and np.array_equal(self.e_grid, e_grid)
                    ):
                        raise ValueError("EOS grids for different species do not match.")
                    self.p_table += weight * p_table
                    self.T_table += weight * T_table
        else:
            with h5py.File(filename, "r") as f:
                if not all(key in f for key in ("rho", "e", "p", "T")):
                    raise ValueError("EOS table is missing required datasets.")
                self.rho_grid = f["rho"][:]
                self.e_grid = f["e"][:]
                self.p_table = f["p"][:]
                self.T_table = f["T"][:]

        if not (
            self.rho_grid.ndim == 1
            and self.e_grid.ndim == 1
            and self.p_table.ndim == 2
            and self.T_table.ndim == 2
        ):
            raise ValueError("EOS table has incorrect dimensions.")

        expected_shape = (len(self.rho_grid), len(self.e_grid))
        if self.p_table.shape != expected_shape or self.T_table.shape != expected_shape:
            raise ValueError("EOS table has inconsistent dimensions.")

        self.p_interp = RegularGridInterpolator((self.rho_grid, self.e_grid), self.p_table)
        self.T_interp = RegularGridInterpolator((self.rho_grid, self.e_grid), self.T_table)

    def pressure(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Interpolate pressure for density ``rho`` and specific energy ``e``."""

        points = np.stack([rho, e], axis=-1)
        return self.p_interp(points)

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Interpolate temperature for density ``rho`` and specific energy ``e``."""

        points = np.stack([rho, e], axis=-1)
        return self.T_interp(points)


def create_eos(model: EOSModel, *, table_path: Path | None = None, gamma: float = 5.0 / 3.0) -> EOSBase:
    """Factory for EOS implementations."""

    if model is EOSModel.IDEAL:
        return IdealGasEOS(gamma=gamma)
    if model is EOSModel.TABULATED and table_path is not None:
        return TabulatedEOS(Path(table_path))
    raise ValueError(f"Unsupported EOS model: {model}")
