from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:  # pragma: no cover - allow running without SciPy
    from scipy.constants import mu_0, pi
except Exception:  # pragma: no cover
    mu_0 = 4e-7 * np.pi
    pi = np.pi


@dataclass
class MZeroInstability:
    """Very simple ``m=0`` (sausage) instability growth model."""

    current: float  # Discharge current [A]
    radius: float  # Pinch radius [m]
    density: float  # Mass density [kg/m^3]

    def growth_rate(self) -> float:
        """Return an order-of-magnitude ``m=0`` growth rate [1/s]."""

        B_theta = mu_0 * self.current / (2 * pi * self.radius)
        return float(abs(B_theta) / np.sqrt(mu_0 * self.density))


__all__ = ["MZeroInstability"]
