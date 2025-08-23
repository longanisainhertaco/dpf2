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

import numpy as np

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


@dataclass
class TabulatedEOS:
    """Very small placeholder for a tabulated SESAME-style EOS.

    The constructor reads a CSV file with three columns ``rho,T,P`` and
    determines a single proportionality constant assuming
    ``P = const * rho * T``.  Real implementations would perform bilinear
    interpolation of the tabulated data; this lightweight approach keeps
    dependencies minimal while exercising the code paths.
    """

    table_path: Path
    cv: float = 1.0

    def __post_init__(self) -> None:
        data = np.loadtxt(self.table_path, delimiter=",", skiprows=1)
        self._const = float(np.mean(data[:, 2] / (data[:, 0] * data[:, 1])))

    def pressure(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        T = self.temperature(rho, e)
        return self._const * rho * T

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return e / self.cv


def create_eos(model: EOSModel, *, table_path: Path | None = None, gamma: float = 5.0 / 3.0) -> EOSBase:
    """Factory for EOS implementations."""

    if model is EOSModel.IDEAL:
        return IdealGasEOS(gamma=gamma)
    if model is EOSModel.TABULATED and table_path is not None:
        return TabulatedEOS(Path(table_path))
    raise ValueError(f"Unsupported EOS model: {model}")
