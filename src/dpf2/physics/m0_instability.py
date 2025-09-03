from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import mu_0, pi
except Exception:  # pragma: no cover
    mu_0 = 4e-7 * np.pi
    pi = np.pi

try:  # pragma: no cover - MPI optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None


@dataclass
class MZeroInstability:
    """Very simple ``m=0`` (sausage) instability growth model."""

    current: np.ndarray | float  # Discharge current [A]
    radius: np.ndarray | float  # Pinch radius [m]
    density: np.ndarray | float  # Mass density [kg/m^3]
    comm: Any | None = None  # MPI communicator

    def growth_rate(self) -> np.ndarray | float:
        """Return an order-of-magnitude ``m=0`` growth rate [1/s]."""

        current = np.asarray(self.current)
        radius = np.asarray(self.radius)
        density = np.asarray(self.density)
        B_theta = mu_0 * current / (2 * pi * radius)
        rate = np.abs(B_theta) / np.sqrt(mu_0 * density)
        if self.comm is not None and MPI is not None:
            gathered = self.comm.allgather(rate)
            rate = np.concatenate([np.atleast_1d(g) for g in gathered], axis=0)
        return rate if rate.ndim > 0 else float(rate)


__all__ = ["MZeroInstability"]
