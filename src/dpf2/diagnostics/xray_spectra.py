from __future__ import annotations
from typing import Iterable, List, Tuple


def compute_xray_spectrum(
    energies: Iterable[float], intensities: Iterable[float], bins: Iterable[float]
) -> Tuple[List[float], List[float]]:
    """Generate an X-ray spectrum from photon energies.

    Parameters
    ----------
    energies:
        Sequence of photon energies in keV.
    intensities:
        Weighting for each photon, typically the photon count.
    bins:
        Bin edges for the resulting spectrum in keV.

    Returns
    -------
    Tuple[List[float], List[float]]
        List of bin centers and the corresponding weighted counts.
    """
    energies = [float(e) for e in energies]
    intensities = [float(i) for i in intensities]
    bins = [float(b) for b in bins]
    if len(energies) != len(intensities):
        raise ValueError("energies and intensities must be the same length")
    if len(bins) < 2:
        raise ValueError("bins must contain at least two edges")
    counts: List[float] = [0.0 for _ in range(len(bins) - 1)]
    for energy, weight in zip(energies, intensities):
        for i in range(len(bins) - 1):
            if bins[i] <= energy < bins[i + 1]:
                counts[i] += weight
                break
    centers = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)]
    return centers, counts
