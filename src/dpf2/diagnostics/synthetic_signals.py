"""Lightweight synthetic diagnostic signal generators.

Each helper operates on an iterable of :class:`~dpf2.core.bases.CouplingState`
objects representing the circuit/plasma coupling at successive time steps.
The functions are intentionally simple and serve as stand-ins for more
specialised diagnostics that would exist in a full application.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence
from pathlib import Path
from bisect import bisect_right
import math

import numpy as np
import h5py

from ..core.bases import CouplingState


def _apply(
    values: List[float],
    response_fn: Callable[[float], float] | None,
    noise_fn: Callable[[float], float] | None,
) -> List[float]:
    if response_fn:
        values = [response_fn(v) for v in values]
    if noise_fn:
        values = [v + noise_fn(v) for v in values]
    return values


def current_waveform(
    history: Iterable[CouplingState],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Return the circuit current for each time step."""

    data = [float(state.current) for state in history]
    return _apply(data, response_fn, noise_fn)


def voltage_waveform(
    history: Iterable[CouplingState],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Return the capacitor voltage for each time step."""

    data = [float(state.voltage) for state in history]
    return _apply(data, response_fn, noise_fn)


def coupled_current_waveform(
    history: Iterable[CouplingState],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Return current including simple back-reaction term.

    The ``CouplingState.back_reaction`` value is interpreted as a voltage
    contribution which is scaled by unity for this synthetic diagnostic.
    """

    data = [float(state.current + state.back_reaction) for state in history]
    return _apply(data, response_fn, noise_fn)


def coupled_voltage_waveform(
    history: Iterable[CouplingState],
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Return voltage including mutual inductance contribution."""

    data = [float(state.voltage + state.mutual_inductance) for state in history]
    return _apply(data, response_fn, noise_fn)


def _load_calibration_hdf5(
    path: str | Path, dataset: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(time, response)`` arrays from an HDF5 calibration file."""

    with h5py.File(path, "r") as fh:
        grp = fh[dataset]
        times = np.array(grp["time"], dtype=float)
        resp = np.array(grp["response"], dtype=float)
    return times, resp


def _apply_instrument_response(
    values: Sequence[float],
    dt: float,
    resp_t: Sequence[float],
    resp_v: Sequence[float],
) -> List[float]:
    """Convolve ``values`` with an impulse response defined by ``resp_t``/``resp_v``."""

    t_grid = np.arange(len(values)) * dt
    impulse = np.interp(t_grid, resp_t, resp_v, left=0.0, right=0.0)
    conv = np.convolve(values, impulse, mode="same")
    return [float(v) for v in conv]


def rogowski_signal(
    history: Iterable[CouplingState],
    dt: float,
    *,
    calibration_file: str | Path | None = None,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Compute a synthetic Rogowski coil signal ``dI/dt``.

    Parameters
    ----------
    calibration_file:
        Optional HDF5 file containing an instrument impulse response under the
        ``"rogowski"`` group with ``time`` and ``response`` datasets.  When
        provided the raw ``dI/dt`` signal is convolved with this response.
    """

    currents = current_waveform(history)
    if len(currents) < 2 or dt <= 0:
        deriv = [0.0 for _ in currents]
    else:
        deriv = [(currents[i + 1] - currents[i]) / dt for i in range(len(currents) - 1)]
        deriv.append(deriv[-1])
    if calibration_file is not None:
        t, r = _load_calibration_hdf5(calibration_file, "rogowski")
        deriv = _apply_instrument_response(deriv, dt, t, r)
    return _apply(deriv, response_fn, noise_fn)


def bdot_signal(
    history: Iterable[CouplingState],
    radius: float,
    dt: float,
    *,
    calibration_file: str | Path | None = None,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Generate a simple B-dot probe signal assuming axial field geometry."""

    mu0 = 4.0 * math.pi * 1e-7
    if radius <= 0:
        raise ValueError("radius must be positive")
    currents = current_waveform(history)
    B = [mu0 * I / (2.0 * math.pi * radius) for I in currents]
    if len(B) < 2 or dt <= 0:
        deriv = [0.0 for _ in B]
    else:
        deriv = [(B[i + 1] - B[i]) / dt for i in range(len(B) - 1)]
        deriv.append(deriv[-1])
    if calibration_file is not None:
        t, r = _load_calibration_hdf5(calibration_file, "bdot")
        deriv = _apply_instrument_response(deriv, dt, t, r)
    return _apply(deriv, response_fn, noise_fn)


def sxr_diode_signal(
    signal: Sequence[float],
    dt: float,
    *,
    calibration_file: str | Path | None = None,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Apply soft X-ray diode filter response to a signal history."""

    data = [float(v) for v in signal]
    if calibration_file is not None:
        t, r = _load_calibration_hdf5(calibration_file, "sxr")
        data = _apply_instrument_response(data, dt, t, r)
    return _apply(data, response_fn, noise_fn)


def neutron_tof_signal(
    energies_MeV: Sequence[float],
    spectrum: Sequence[float],
    flight_path_m: float,
    time_bins_s: Sequence[float],
    *,
    calibration_file: str | Path | None = None,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Generate a neutron time-of-flight detector signal.

    ``energies_MeV`` and ``spectrum`` provide the neutron energy distribution.
    ``flight_path_m`` is the source-to-detector distance.  ``time_bins_s``
    defines the histogram bins for the TOF signal.
    """

    if len(energies_MeV) != len(spectrum):
        raise ValueError("energies_MeV and spectrum must have the same length")
    if any(time_bins_s[i] >= time_bins_s[i + 1] for i in range(len(time_bins_s) - 1)):
        raise ValueError("time_bins_s must be monotonically increasing")

    m_n = 1.67492749804e-27  # neutron mass [kg]
    e_j = [e * 1.602176634e-13 for e in energies_MeV]  # MeV → J
    speeds = [math.sqrt(2.0 * E / m_n) for E in e_j]
    times = [flight_path_m / v for v in speeds]
    hist = [0.0 for _ in range(len(time_bins_s) - 1)]
    for t, count in zip(times, spectrum):
        idx = bisect_right(time_bins_s, t) - 1
        if 0 <= idx < len(hist):
            hist[idx] += float(count)

    if calibration_file is not None:
        dt = time_bins_s[1] - time_bins_s[0] if len(time_bins_s) > 1 else 1.0
        t_resp, r_resp = _load_calibration_hdf5(calibration_file, "tof")
        hist = _apply_instrument_response(hist, dt, t_resp, r_resp)
    return _apply(hist, response_fn, noise_fn)


__all__ = [
    "current_waveform",
    "voltage_waveform",
    "coupled_current_waveform",
    "coupled_voltage_waveform",
    "rogowski_signal",
    "bdot_signal",
    "sxr_diode_signal",
    "neutron_tof_signal",
]

