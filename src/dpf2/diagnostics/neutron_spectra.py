from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from typing import Sequence, List
import math

# Neutron mass used for simple time-of-flight calculations (kg)
M_N = 1.674e-27


@dataclass
class Detector:
    """Simple representation of a neutron detector."""

    angle_deg: float
    distance_m: float
    name: str


@dataclass
class DetectorLayout:
    """Container describing a collection of detectors on a ring."""

    angles: Sequence[float]
    distance_m: float
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.distance_m <= 0:
            raise ValueError("distance_m must be positive")
        if self.names and len(self.names) != len(self.angles):
            raise ValueError("names must match number of angles")
        self.detectors: List[Detector] = []
        for i, ang in enumerate(self.angles):
            name = self.names[i] if self.names else f"detector_{i}"
            self.detectors.append(Detector(float(ang), float(self.distance_m), name))

    def angles_deg(self) -> List[float]:
        """Return detector viewing angles in degrees."""

        return [d.angle_deg for d in self.detectors]

    def names_list(self) -> List[str]:
        """Return detector names in layout order."""

        return [d.name for d in self.detectors]


def synthetic_tof_spectrum(
    energies: Sequence[float],
    flux: Sequence[float],
    distance: float,
    time_bins: Sequence[float],
    m_n: float = M_N,
) -> List[float]:
    """Generate a simple neutron time-of-flight histogram.

    Parameters
    ----------
    energies:
        Monotonically increasing energy grid in joules.
    flux:
        Differential flux corresponding to ``energies``.
    distance:
        Source-to-detector distance in meters.
    time_bins:
        Edges of time-of-flight histogram bins in seconds.
    m_n:
        Neutron mass in kg.  Defaults to :data:`M_N`.

    Returns
    -------
    list of float
        Bin-integrated counts for the requested histogram.
    """

    if len(energies) != len(flux):
        raise ValueError("energies and flux must be same length")
    if any(time_bins[i] >= time_bins[i + 1] for i in range(len(time_bins) - 1)):
        raise ValueError("time_bins must be monotonically increasing")
    hist = [0.0 for _ in range(len(time_bins) - 1)]
    for e1, e2, f1, f2 in zip(energies[:-1], energies[1:], flux[:-1], flux[1:]):
        dE = e2 - e1
        contrib = 0.5 * (f1 + f2) * dE
        e_mid = 0.5 * (e1 + e2)
        t = distance / math.sqrt(2.0 * e_mid / m_n)
        idx = bisect_right(time_bins, t) - 1
        if 0 <= idx < len(hist):
            hist[idx] += contrib
    return hist


def angular_spectrum(
    angles: Sequence[float],
    base_yield: float,
    anisotropy: float = 0.0,
) -> List[float]:
    """Create a simple angular yield spectrum using a cosine model."""

    spectrum: List[float] = []
    for ang in angles:
        val = base_yield * (1.0 + anisotropy * math.cos(math.radians(ang)))
        spectrum.append(float(val))
    return spectrum


def anisotropy_metric(values: Sequence[float]) -> float:
    """Return a rudimentary anisotropy metric ``(max - min) / mean``."""

    if not values:
        return 0.0
    mean = sum(float(v) for v in values) / len(values)
    if mean == 0.0:
        return 0.0
    return (max(values) - min(values)) / mean


__all__ = [
    "Detector",
    "DetectorLayout",
    "synthetic_tof_spectrum",
    "angular_spectrum",
    "anisotropy_metric",
]
