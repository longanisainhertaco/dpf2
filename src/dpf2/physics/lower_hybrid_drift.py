from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np

try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import e, m_e, m_p
except Exception:  # pragma: no cover
    e = 1.602176634e-19
    m_e = 9.10938356e-31
    m_p = 1.67262192369e-27


def _to_array(val: Any) -> np.ndarray:
    """Return ``val`` as a floating point array.

    Mirrors :func:`_to_array` from ``m0_instability`` and tolerates the light
    weight ``numpy`` substitute used in the tests.
    """

    try:  # pragma: no cover - real NumPy path
        return np.asarray(val, dtype=float)
    except TypeError:  # pragma: no cover - ``numpy_stub`` path
        return np.asarray(val)


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

    def growth_rate(self, k: Any) -> np.ndarray:
        ks = _to_array(k)
        omega_lh = self.frequency()
        rates = 0.1 * omega_lh * np.exp(-(ks * ks))
        return rates

    def evolve(self, amplitude: Any, k: Any, dt: float):
        amp = _to_array(amplitude)
        rate = _to_array(self.growth_rate(k))
        evolved = amp * np.exp(np.clip(rate * dt, -50.0, 50.0))
        return evolved


__all__ = ["LowerHybridDrift"]
