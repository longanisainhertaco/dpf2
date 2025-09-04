from __future__ import annotations

"""Parameterized geometry classes for simple electrode shapes."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TaperedGeometry:
    """Simple tapered column geometry.

    Parameters
    ----------
    length: float
        Total length of the column [m].
    r_base: float
        Base radius at ``z=0`` [m].
    r_top: float
        Radius at the top of the column [m].
    """

    length: float
    r_base: float
    r_top: float

    def radius_profile(self, n: int = 50) -> List[Tuple[float, float]]:
        """Return ``(z, r)`` pairs describing the taper."""
        if n < 2:  # pragma: no cover - input validation
            raise ValueError("n must be >= 2")
        dz = self.length / (n - 1)
        dr = (self.r_top - self.r_base) / (n - 1)
        return [(i * dz, self.r_base + i * dr) for i in range(n)]


@dataclass
class HollowGeometry:
    """Cylindrical geometry with an inner bore.

    Parameters
    ----------
    length: float
        Length of the cylinder [m].
    r_outer: float
        Outer radius [m].
    r_inner: float
        Inner radius (hollow region) [m].
    """

    length: float
    r_outer: float
    r_inner: float

    def volume(self) -> float:
        """Return the volume of material."""
        import math

        return math.pi * self.length * (self.r_outer**2 - self.r_inner**2)


@dataclass
class ReentrantGeometry:
    """Re-entrant cavity geometry defined by straight segments.

    Parameters
    ----------
    segments: list of ``(z, r)`` pairs
        Describes the cavity profile along the axis.
    """

    segments: List[Tuple[float, float]]

    def profile(self) -> List[Tuple[float, float]]:
        """Return the profile coordinates."""
        return list(self.segments)
