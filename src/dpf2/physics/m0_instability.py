from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np

try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import mu_0, pi
except Exception:  # pragma: no cover
    mu_0 = 4e-7 * math.pi
    pi = math.pi

try:  # pragma: no cover - MPI optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None


def _to_array(val: Any) -> np.ndarray:
    """Return ``val`` as a floating point array.

    ``numpy_stub`` used in the tests provides :func:`asarray` without support
    for the ``dtype`` keyword.  The helper therefore attempts a typed
    conversion but gracefully falls back when running with the stub
    implementation.
    """

    try:  # pragma: no cover - exercised when real NumPy is present
        return np.asarray(val, dtype=float)
    except TypeError:  # pragma: no cover - ``numpy_stub`` path
        return np.asarray(val)


@dataclass
class MZeroInstability:
    """Very simple ``m=0`` (sausage) instability growth model."""

    current: Any  # Discharge current [A]
    radius: Any  # Pinch radius [m]
    density: Any  # Mass density [kg/m^3]
    comm: Any | None = None  # MPI communicator

    def growth_rate(self) -> np.ndarray:
        curr = _to_array(self.current)
        rad = _to_array(self.radius)
        dens = _to_array(self.density)
        rate = np.abs(mu_0 * curr / (2 * pi * rad)) / np.sqrt(mu_0 * dens)
        if self.comm is not None and MPI is not None:
            gathered = self.comm.allgather(rate)
            rate = np.concatenate([np.asarray(g, dtype=float) for g in gathered], axis=0)
        return rate

    def evolve(self, amplitude: Any, dt: float):
        amp = _to_array(amplitude)
        rate = _to_array(self.growth_rate())
        evolved = amp * np.exp(np.clip(rate * dt, -50.0, 50.0))
        return evolved


__all__ = ["MZeroInstability"]
