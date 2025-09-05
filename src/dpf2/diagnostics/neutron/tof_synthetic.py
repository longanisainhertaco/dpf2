from __future__ import annotations

import math
from typing import Sequence, List, Tuple


def cross_correlate(a: Sequence[float], b: Sequence[float], dt: float) -> Tuple[List[float], List[float]]:
    """Return cross-correlation of sequences ``a`` and ``b``.

    Parameters
    ----------
    a, b:
        Input sequences.  They need not be the same length; shorter sequences
        are zero padded.
    dt:
        Sampling interval in seconds used to scale the lag values.
    """
    n = max(len(a), len(b))
    if n == 0:
        return [], []
    pa = list(a) + [0.0] * (n - len(a))
    pb = list(b) + [0.0] * (n - len(b))
    mean_a = sum(pa) / n
    mean_b = sum(pb) / n
    corr: List[float] = []
    lags: List[float] = []
    for lag in range(-n + 1, n):
        val = 0.0
        for i in range(n):
            j = i - lag
            if 0 <= j < n:
                val += (pa[i] - mean_a) * (pb[j] - mean_b)
        corr.append(val)
        lags.append(lag * dt)
    return lags, corr


def cross_correlation_with_iv(
    counts: Sequence[float], current: Sequence[float], voltage: Sequence[float], dt: float
) -> Tuple[List[float], List[float]]:
    """Cross-correlate ``counts`` with the ``I*V`` power history.

    The instantaneous electrical power ``abs(I*V)`` is used for the correlation.
    """
    power = [abs(float(i) * float(v)) for i, v in zip(current, voltage)]
    padded_power = power + [0.0] * (len(counts) - len(power))
    return cross_correlate(counts, padded_power, dt)


def synthetic_tof_from_iv(
    current: Sequence[float],
    voltage: Sequence[float],
    dt: float,
    distance_m: float,
    energies_mev: Sequence[float],
    *,
    time_offset: float = 0.0,
    align_peaks: bool = False,
) -> Tuple[List[float], List[float]]:
    """Generate a synthetic neutron time-of-flight signal from I–V traces.

    Parameters
    ----------
    current, voltage:
        Sequences representing the current and voltage waveforms.  Both must be
        the same length and sampled at ``dt`` seconds.
    dt:
        Sampling interval in seconds.
    distance_m:
        Source-to-detector distance in metres.
    energies_mev:
        Iterable of neutron energies in MeV used to compute time-of-flight
        delays.
    time_offset:
        Additional delay applied to the generated signal, in seconds.
    align_peaks:
        When ``True`` the resulting signal is cross-correlated with the
        ``I*V`` product and shifted so that the correlation peak occurs at
        zero lag.  This is useful for verifying timing alignment.
    """
    if len(current) != len(voltage):
        raise ValueError("current and voltage must be the same length")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not energies_mev:
        return [], []

    m_n = 1.67492749804e-27  # neutron mass [kg]
    tofs: List[float] = []
    for e in energies_mev:
        e_j = float(e) * 1.602176634e-13
        v = math.sqrt(2.0 * e_j / m_n)
        tofs.append(distance_m / v)

    offset_samples = int(round(time_offset / dt))
    max_tof = max(tofs)
    total = len(current) + int(math.ceil(max_tof / dt)) + abs(offset_samples) + 1
    counts = [0.0] * total
    power = [abs(float(i) * float(v)) for i, v in zip(current, voltage)]
    for i, amp in enumerate(power):
        for tof in tofs:
            idx = i + int(round(tof / dt)) + offset_samples
            if 0 <= idx < total:
                counts[idx] += amp
    times = [i * dt for i in range(total)]

    if align_peaks and any(power) and any(counts):
        padded_power = power + [0.0] * (len(counts) - len(power))
        lags, corr = cross_correlate(counts, padded_power, dt)
        best_lag = lags[corr.index(max(corr))]
        shift = int(round(best_lag / dt))
        if shift > 0:
            counts = counts[shift:] + [0.0] * shift
        elif shift < 0:
            counts = [0.0] * (-shift) + counts[:shift]
        times = [t - best_lag for t in times]
    return times, counts

__all__ = ["synthetic_tof_from_iv", "cross_correlation_with_iv", "cross_correlate"]
