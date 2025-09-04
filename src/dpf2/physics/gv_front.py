from __future__ import annotations

"""Simplified Gratton--Vargas sheath front model.

The Gratton--Vargas (GV) snowplow model provides an analytic description of
how a plasma current sheath propagates in the radial--axial plane of a coaxial
accelerator.  For lightweight benchmarking and educational purposes we model
the sheath as a semicircular arc of radius equal to the anode radius ``a`` that
moves axially with constant velocity ``v``.  While greatly simplified compared
with the full GV solution, this captures the expected parabolic ``r(z)`` shape
and provides closed‑form arrival times.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class GVFront:
    """Model the r--z current sheath front.

    Parameters
    ----------
    anode_radius:
        Anode radius ``a`` in metres.
    velocity:
        Axial propagation speed ``v`` of the sheath in metres per second.
    """

    anode_radius: float
    velocity: float

    def radius(self, z: float | Iterable[float]):
        """Return radial position of the sheath front at axial coordinate ``z``.

        The simplified model assumes the front traces a semicircle of radius
        ``a``.  Values of ``z`` outside ``[0, a]`` are clipped to the physical
        domain.
        """

        if isinstance(z, Iterable) and not isinstance(z, (float, int)):
            vals = []
            for val in z:
                v = max(0.0, min(float(val), self.anode_radius))
                diff = self.anode_radius**2 - v**2
                vals.append(diff**0.5 if diff > 0 else 0.0)
            return np.array(vals)
        else:
            z_val = max(0.0, min(float(z), self.anode_radius))
            diff = self.anode_radius**2 - z_val**2
            return diff**0.5 if diff > 0 else 0.0

    def arrival_time(self, z: float | Iterable[float]):
        """Return the time for the sheath to reach axial position ``z``."""

        if self.velocity <= 0:
            raise ValueError("velocity must be positive")
        if isinstance(z, Iterable) and not isinstance(z, (float, int)):
            return np.array([float(val) / self.velocity for val in z])
        else:
            return float(z) / self.velocity


__all__ = ["GVFront"]
