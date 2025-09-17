import logging
from typing import Iterable, Dict, Any

import numpy as np

SAUSAGE_GROWTH_THRESHOLD = 0.05
KINK_GROWTH_THRESHOLD = 0.05


def fft_m_modes(field: np.ndarray, theta_axis: int = -1) -> np.ndarray:
    """Return absolute FFT amplitudes of azimuthal modes.

    Parameters
    ----------
    field: np.ndarray
        Field sampled on uniform azimuthal grid. The azimuthal axis is
        specified by ``theta_axis``.
    theta_axis: int
        Axis corresponding to the azimuthal direction.

    Returns
    -------
    np.ndarray
        Absolute value of Fourier coefficients along ``theta_axis``
        normalised by the number of azimuthal samples.
    """
    field = np.asarray(field)
    n_theta = field.shape[theta_axis]
    fft = np.fft.fft(field, axis=theta_axis)
    return np.abs(fft) / n_theta


def growth_rate(
    time_series: Iterable[np.ndarray], dt: float, m: int, theta_axis: int = -1
) -> float:
    """Estimate exponential growth rate for mode ``m``.

    Parameters
    ----------
    time_series: iterable of np.ndarray
        Sequence of field snapshots ordered in time.
    dt: float
        Time separation between consecutive snapshots.
    m: int
        Mode number whose growth rate is requested.
    theta_axis: int
        Axis corresponding to the azimuthal direction.
    """
    amplitudes = []
    for snap in time_series:
        modes = fft_m_modes(snap, theta_axis)
        amplitudes.append(np.mean(modes[..., m]))
    amplitudes = np.asarray(amplitudes)
    amplitudes = np.where(amplitudes <= 0, 1e-12, amplitudes)
    times = np.arange(len(amplitudes)) * dt
    slope, _ = np.polyfit(times, np.log(amplitudes), 1)
    return float(slope)


def analyze_instabilities(
    time_series: Iterable[np.ndarray],
    dt: float,
    ez_series: Iterable[float] | None = None,
    thresholds: Dict[str, float] | None = None,
    theta_axis: int = -1,
) -> Dict[str, Any]:
    """Analyse m=0 and m=1 modes and detect instability onsets.

    Parameters
    ----------
    time_series: iterable of np.ndarray
        Field snapshots ordered in time.
    dt: float
        Time step between snapshots.
    ez_series: iterable of float, optional
        On-axis electric field evolution used to relate mode growth to
        axial surges and beam formation.
    thresholds: dict, optional
        Mapping of ``{"sausage": value, "kink": value}`` representing
        growth-rate thresholds that trigger alerts.
    theta_axis: int
        Axis corresponding to azimuthal direction.

    Returns
    -------
    dict
        Dictionary with growth rates, optional Ez surge information and
        list of instabilities whose thresholds were exceeded.
    """
    if thresholds is None:
        thresholds = {
            "sausage": SAUSAGE_GROWTH_THRESHOLD,
            "kink": KINK_GROWTH_THRESHOLD,
        }

    gr0 = growth_rate(time_series, dt, 0, theta_axis)
    gr1 = growth_rate(time_series, dt, 1, theta_axis)

    alerts = []
    if gr0 > thresholds.get("sausage", float("inf")):
        alerts.append("sausage")
        logging.warning("Sausage instability threshold exceeded")
    if gr1 > thresholds.get("kink", float("inf")):
        alerts.append("kink")
        logging.warning("Kink instability threshold exceeded")

    result: Dict[str, Any] = {
        "growth_rates": {"m0": gr0, "m1": gr1},
        "alerts": alerts,
    }

    if ez_series is not None:
        ez_series = np.asarray(ez_series)
        derivative = np.gradient(ez_series) / dt
        surge_index = int(np.argmax(derivative))
        result["ez_surge_time"] = surge_index * dt
        if "kink" in alerts:
            result["beam_onset_time"] = surge_index * dt
    return result
