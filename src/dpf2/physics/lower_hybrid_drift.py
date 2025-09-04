from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import e, m_e, m_p
except Exception:  # pragma: no cover
    e = 1.602176634e-19
    m_e = 9.10938356e-31
    m_p = 1.67262192369e-27


def _to_seq(val) -> list[float]:
    if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        return [float(v) for v in val]
    return [float(val)]


@dataclass
class LowerHybridDrift:
    """Minimal lower-hybrid drift instability model."""

    B: float  # Magnetic field strength [T]
    n_i: float  # Ion number density [m^-3]
    m_i: float = m_p  # Ion mass [kg]

    def frequency(self) -> float:
        omega_ci = e * self.B / self.m_i
        omega_ce = e * self.B / m_e
        return abs(omega_ci * omega_ce) ** 0.5

    def growth_rate(self, k):
        ks = _to_seq(k)
        omega_lh = self.frequency()
        rates = [0.1 * omega_lh * math.exp(-(kk * kk)) for kk in ks]
        return rates if len(rates) > 1 else rates[0]

    def evolve(self, amplitude, k, dt: float):
        amps = _to_seq(amplitude)
        rates = _to_seq(self.growth_rate(k))
        evolved = [a * math.exp(max(min(r * dt, 50.0), -50.0)) for a, r in zip(amps, rates)]
        return evolved if len(evolved) > 1 else evolved[0]


__all__ = ["LowerHybridDrift"]
