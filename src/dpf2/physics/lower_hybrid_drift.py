from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import math

try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import e, m_e, m_p
except Exception:  # pragma: no cover
    e = 1.602176634e-19
    m_e = 9.10938356e-31
    m_p = 1.67262192369e-27


@dataclass
class LowerHybridDrift:
    """Minimal lower-hybrid drift instability model.

    The model provides a very small subset of lower-hybrid physics that is
    sufficient for regression tests.  It estimates the characteristic
    frequency of the mode and a simple exponential growth rate used by
    reduced transport closures.
    """

    B: float  # Magnetic field strength [T]
    n_i: float  # Ion number density [m^-3]
    m_i: float = m_p  # Ion mass [kg]

    def frequency(self) -> float:
        """Return the lower-hybrid frequency [rad/s]."""

        omega_ci = e * self.B / self.m_i
        omega_ce = e * self.B / m_e
        return float(np.sqrt(abs(omega_ci * omega_ce)))

    def growth_rate(self, k: float) -> float:
        """Crude exponential growth rate for a given wavenumber ``k``."""

        omega_lh = self.frequency()
        return float(0.1 * omega_lh * math.exp(-k**2))


__all__ = ["LowerHybridDrift"]
