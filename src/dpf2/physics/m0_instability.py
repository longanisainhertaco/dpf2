from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import math

try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import mu_0, pi
except Exception:  # pragma: no cover
    mu_0 = 4e-7 * math.pi
    pi = math.pi

try:  # pragma: no cover - MPI optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None


def _to_seq(val: Any) -> list[float]:
    if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        return [float(v) for v in val]
    return [float(val)]


@dataclass
class MZeroInstability:
    """Very simple ``m=0`` (sausage) instability growth model."""

    current: Any  # Discharge current [A]
    radius: Any  # Pinch radius [m]
    density: Any  # Mass density [kg/m^3]
    comm: Any | None = None  # MPI communicator

    def growth_rate(self):
        curr = _to_seq(self.current)
        rad = _to_seq(self.radius)
        dens = _to_seq(self.density)
        rates = [abs(mu_0 * c / (2 * pi * r)) / math.sqrt(mu_0 * d) for c, r, d in zip(curr, rad, dens)]
        if self.comm is not None and MPI is not None:
            gathered = self.comm.allgather(rates)
            rates = [r for g in gathered for r in g]
        return rates if len(rates) > 1 else rates[0]

    def evolve(self, amplitude, dt: float):
        amps = _to_seq(amplitude)
        rates = _to_seq(self.growth_rate())
        evolved = [a * math.exp(max(min(r * dt, 50.0), -50.0)) for a, r in zip(amps, rates)]
        return evolved if len(evolved) > 1 else evolved[0]


__all__ = ["MZeroInstability"]
