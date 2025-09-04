"""Neutron diagnostic utilities for angular spectra and time-of-flight."""

from __future__ import annotations

from typing import Sequence, Callable, Dict, List

from ..neutron_yield_model import IonBeamEDF, compute_directional_spectrum
from .neutron_yield import compute_beam_target_yield


def angular_spectrum(
    ion_edf: IonBeamEDF,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    energy_bins: Sequence[float],
) -> List[List[float]]:
    """Return per-angle energy spectra using :func:`compute_directional_spectrum`."""

    return compute_directional_spectrum(ion_edf, cross_section, angles, energy_bins)


def synthesize_tof(
    ion_edf: IonBeamEDF,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    distance: float,
    time_bins: Sequence[float],
) -> Dict[str, List[float]]:
    """Generate synthetic time-of-flight histograms for a detector layout."""

    _yields, tofs = compute_beam_target_yield(
        ion_edf, cross_section, angles, distance, time_bins
    )
    return {f"detector_{i}": hist for i, hist in enumerate(tofs)}


def compute_anisotropy(
    yields: Sequence[float], metric: str = "max_min_over_mean"
) -> float:
    """Compute anisotropy metric from per-angle yields."""

    vals = [float(v) for v in yields]
    if not vals:
        return 0.0
    if metric == "forward_backward_ratio":
        half = len(vals) // 2
        forward = sum(vals[:half])
        backward = sum(vals[half:])
        return forward / backward if backward > 0 else 0.0
    mean = sum(vals) / len(vals)
    if mean == 0:
        return 0.0
    return (max(vals) - min(vals)) / mean


__all__ = ["angular_spectrum", "synthesize_tof", "compute_anisotropy"]

