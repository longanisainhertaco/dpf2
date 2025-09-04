"""Synthetic time-of-flight utilities."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..neutron_spectra import (
    synthetic_tof_spectrum,
    correlate_tof_peaks_with_circuit_iv,
)
from .angular import _apply_optional_response


def synthetic_tof_correlated(
    energies: Sequence[float],
    flux: Sequence[float],
    distance: float,
    time_bins: Sequence[float],
    circuit_time: Sequence[float],
    current: Sequence[float],
    voltage: Sequence[float],
    response: Dict[str, float] | None = None,
) -> Tuple[List[float], List[Tuple[float, float]], float]:
    """Return ToF histogram and correlation with circuit ``I``–``V`` traces."""

    hist = synthetic_tof_spectrum(energies, flux, distance, time_bins)
    mid = [0.5 * (time_bins[i] + time_bins[i + 1]) for i in range(len(time_bins) - 1)]
    if response:
        hist = _apply_optional_response(mid, hist, response)
    peaks, _lags, _corr, max_lag = correlate_tof_peaks_with_circuit_iv(
        time_bins, hist, circuit_time, current, voltage
    )
    return hist, peaks, max_lag


__all__ = ["synthetic_tof_correlated"]
