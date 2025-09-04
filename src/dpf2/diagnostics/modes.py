"""Fourier mode decomposition utilities.

This module provides lightweight helpers for analysing azimuthal
perturbations in cylindrical data sets.  The functions operate on both
2-D ``(r,\theta)`` and 3-D ``(r,\theta,z)`` fields where the azimuthal
angle is assumed to be along a dedicated axis.  The primary entry point
is :func:`azimuthal_mode_spectrum` which returns the magnitude of each
mode ``m`` of the supplied field.  A small convenience routine for
estimating exponential growth rates of modal amplitudes is also
included.
"""

from __future__ import annotations

import math
from itertools import product
import numpy as np
from typing import Sequence

__all__ = [
    "azimuthal_mode_spectrum",
    "azimuthal_decomposition",
    "growth_rate",
    "lh_azimuthal_power",
    "log_impedance_ratio",
]


def azimuthal_mode_spectrum(field: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the azimuthal Fourier mode spectrum of ``field``.

    Parameters
    ----------
    field:
        Input array containing the scalar quantity to analyse.  The
        azimuthal direction is assumed to be aligned with ``axis``.
    axis:
        Axis corresponding to the angular coordinate.  Defaults to the
        last axis which matches the representation used in many tests.

    Returns
    -------
    ndarray
        One-dimensional array containing the averaged amplitude of each
        azimuthal mode ``m`` present in ``field``.  The DC component
        (``m=0``) is returned as-is while all other modes are scaled such
        that a pure ``cos(m\theta)`` signal has amplitude ``1``.
    """

    data = np.asarray(field)

    # ``numpy`` may not provide FFT functionality in the lightweight stub used
    # by the tests.  When ``np.fft`` is unavailable a very small discrete
    # transform is computed explicitly in Python.  The implementation only
    # supports analysing along the last axis in this fallback mode which is
    # sufficient for the tests.
    if hasattr(np, "fft"):
        data = np.moveaxis(data, axis, -1)
        n = data.shape[-1]
        if n == 0:
            return np.array([])
        coeff = np.fft.rfft(data, axis=-1) / float(n)
        amp = np.abs(coeff)
        if n % 2 == 0:
            if amp.shape[-1] > 2:
                amp[..., 1:-1] *= 2.0
        else:
            if amp.shape[-1] > 1:
                amp[..., 1:] *= 2.0
        mean_axes = tuple(range(amp.ndim - 1))
        return np.mean(amp, axis=mean_axes)

    # Fallback manual decomposition for environments without ``np.fft``.
    if axis not in (-1, data.ndim - 1):
        raise ValueError("manual mode spectrum only supports the last axis")
    n = data.shape[-1]
    if n == 0:
        return np.array([])
    theta = [2.0 * math.pi * k / n for k in range(n)]
    m_max = n // 2
    spectrum = []
    # Iterate over modes
    for m in range(m_max + 1):
        c_sum = 0.0
        count = 0
        cos_vals = [math.cos(m * t) for t in theta]
        for idx in product(*(range(s) for s in data.shape[:-1])):
            row = data[idx]
            val = 0.0
            for j in range(n):
                val += row[j] * cos_vals[j]
            val = val / n
            if m != 0:
                val *= 2.0
            c_sum += abs(val)
            count += 1
        spectrum.append(c_sum / count if count else 0.0)
    return np.array(spectrum)


def azimuthal_decomposition(field: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return complex azimuthal Fourier coefficients of ``field``.

    This is similar to :func:`azimuthal_mode_spectrum` but retains the
    complex phase information of each mode.  The coefficients are scaled so
    that a pure ``cos(m\theta)`` signal yields a real coefficient of one and
    a vanishing imaginary part.
    """

    data = np.asarray(field)

    if hasattr(np, "fft"):
        data = np.moveaxis(data, axis, -1)
        n = data.shape[-1]
        if n == 0:
            return np.array([])
        coeff = np.fft.rfft(data, axis=-1) / float(n)
        if n % 2 == 0:
            if coeff.shape[-1] > 2:
                coeff[..., 1:-1] *= 2.0
        else:
            if coeff.shape[-1] > 1:
                coeff[..., 1:] *= 2.0
        mean_axes = tuple(range(coeff.ndim - 1))
        return np.mean(coeff, axis=mean_axes)

    if axis not in (-1, data.ndim - 1):
        raise ValueError("manual mode decomposition only supports the last axis")
    n = data.shape[-1]
    if n == 0:
        return np.array([])
    theta = [2.0 * math.pi * k / n for k in range(n)]
    m_max = n // 2
    coeff = []
    cos_cache = [[math.cos(m * t) for t in theta] for m in range(m_max + 1)]
    sin_cache = [[math.sin(m * t) for t in theta] for m in range(m_max + 1)]
    for m in range(m_max + 1):
        c_sum = 0.0 + 0.0j
        count = 0
        for idx in product(*(range(s) for s in data.shape[:-1])):
            row = data[idx]
            a = 0.0
            b = 0.0
            for j in range(n):
                a += row[j] * cos_cache[m][j]
                b += row[j] * sin_cache[m][j]
            a /= n
            b /= n
            if m != 0:
                a *= 2.0
                b *= 2.0
            c_sum += complex(a, -b)
            count += 1
        coeff.append(c_sum / count if count else 0.0)
    return np.asarray(coeff)


def growth_rate(previous: Sequence[float], current: Sequence[float], dt: float) -> np.ndarray:
    """Estimate exponential growth rates between two spectra.

    The growth rate for each mode is computed using ``ln(A1/A0) / dt``
    where ``A0`` and ``A1`` are the modal amplitudes at two successive
    times.  Modes with vanishing initial amplitude yield a growth rate of
    ``0``.

    Parameters
    ----------
    previous, current:
        Modal amplitudes at two different times.
    dt:
        Time step separating the samples.
    """

    a0 = np.asarray(previous)
    a1 = np.asarray(current)
    rate = []
    for p, c in zip(a0, a1):
        if p > 0:
            rate.append(math.log(c / p) / float(dt))
        else:
            rate.append(0.0)
    return np.asarray(rate)


# ---------------------------------------------------------------------------
def lh_azimuthal_power(field: np.ndarray, omega_lh: float, axis: int = -1) -> float:
    """Return azimuthal power near the lower-hybrid frequency.

    The function computes the azimuthal mode spectrum of ``field`` and
    selects the mode whose index most closely matches ``omega_lh``.  The
    amplitude of this mode serves as a crude proxy for the power contained in
    lower-hybrid drift waves.
    """

    spectrum = azimuthal_mode_spectrum(field, axis=axis)
    if len(spectrum) == 0:
        return 0.0
    m = int(round(omega_lh)) % len(spectrum)
    return float(spectrum[m])


def log_impedance_ratio(
    eta_plasma: np.ndarray,
    ne: np.ndarray,
    Te: np.ndarray,
    Z: float | np.ndarray,
) -> np.ndarray:
    """Logarithmic plasma impedance relative to Spitzer prediction.

    ``eta_plasma`` is compared against the classical Spitzer resistivity and
    the base-10 logarithm of the ratio is returned.  The helper is designed
    for lightweight diagnostics and therefore accepts array-like inputs for
    convenience.
    """

    from dpf2.hall_mhd_solver import spitzer_resistivity

    eta_s = spitzer_resistivity(ne, Te, Z)
    eta_p = np.asarray(eta_plasma)
    return np.log10(np.maximum(eta_p, 1e-30) / np.maximum(eta_s, 1e-30))
