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
import json
import math

import numpy as np
import h5py

try:
    from ..dpf_config import DPFConfig  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DPFConfig = None  # type: ignore

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


def _load_calibration_curve(
    path: str | Path, dataset: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(time, response)`` arrays from a calibration file."""

    p = Path(path)
    if p.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(p, "r") as fh:
            grp = fh[dataset]
            times = np.array(grp["time"], dtype=float)
            resp = np.array(grp["response"], dtype=float)
        return times, resp
    data = json.loads(p.read_text())
    scale = float(data.get("scale", 1.0))
    return np.array([0.0]), np.array([scale])


def _apply_instrument_response(
    values: Sequence[float],
    dt: float,
    resp_t: Sequence[float],
    resp_v: Sequence[float],
) -> List[float]:
    """Convolve ``values`` with an impulse response defined by ``resp_t``/``resp_v``."""

    t_grid = np.arange(len(values)) * dt
    try:
        impulse = np.interp(t_grid, resp_t, resp_v, left=0.0, right=0.0)
    except TypeError:  # pragma: no cover - for minimal numpy stubs
        impulse = []
        for t in t_grid:
            if t < resp_t[0] or t > resp_t[-1]:
                impulse.append(0.0)
            elif len(resp_t) == 1:
                impulse.append(resp_v[0])
            else:  # fallback interpolation without bounds
                impulse.append(np.interp(t, resp_t, resp_v))
    try:
        conv = np.convolve(values, impulse, mode="same")
    except AttributeError:  # pragma: no cover - for minimal numpy stubs
        n = len(values)
        m = len(impulse)
        full = [0.0] * (n + m - 1)
        for i, a in enumerate(values):
            for j, b in enumerate(impulse):
                full[i + j] += a * b
        start = (m - 1) // 2
        conv = full[start : start + n]
    return [float(v) for v in conv]


def _cfg_calibration(cfg: "DPFConfig | None", attr: str) -> Path | None:
    """Return calibration path for ``attr`` from :class:`DPFConfig`.

    Parameters
    ----------
    cfg:
        Optional configuration object providing a ``diagnostics`` section.
    attr:
        Attribute name on ``cfg.diagnostics`` holding the calibration path.
    """

    if cfg is None or DPFConfig is None:
        return None
    diag = getattr(cfg, "diagnostics", None)
    if diag is None:
        return None
    path = getattr(diag, attr, None)
    if path is None:
        return None
    return Path(path)


def rogowski_signal(
    history: Iterable[CouplingState],
    dt: float,
    *,
    cfg: "DPFConfig | None" = None,
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
    if calibration_file is None:
        calibration_file = _cfg_calibration(cfg, "rogowski_calibration_path")
    if calibration_file is not None:
        t, r = _load_calibration_curve(calibration_file, "rogowski")
        deriv = _apply_instrument_response(deriv, dt, t, r)
    return _apply(deriv, response_fn, noise_fn)


def bdot_signal(
    history: Iterable[CouplingState],
    radius: float,
    dt: float,
    *,
    cfg: "DPFConfig | None" = None,
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
    if calibration_file is None:
        calibration_file = _cfg_calibration(cfg, "bdot_calibration_path")
    if calibration_file is not None:
        t, r = _load_calibration_curve(calibration_file, "bdot")
        deriv = _apply_instrument_response(deriv, dt, t, r)
    return _apply(deriv, response_fn, noise_fn)


def sxr_diode_signal(
    signal: Sequence[float],
    dt: float,
    *,
    cfg: "DPFConfig | None" = None,
    calibration_file: str | Path | None = None,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[float]:
    """Apply soft X-ray diode filter response to a signal history."""

    data = [float(v) for v in signal]
    if calibration_file is None:
        calibration_file = _cfg_calibration(cfg, "sxr_calibration_path")
    if calibration_file is not None:
        t, r = _load_calibration_curve(calibration_file, "sxr")
        data = _apply_instrument_response(data, dt, t, r)
    return _apply(data, response_fn, noise_fn)


def neutron_tof_signal(
    energies_MeV: Sequence[float],
    spectrum: Sequence[float],
    flight_path_m: float,
    time_bins_s: Sequence[float],
    *,
    cfg: "DPFConfig | None" = None,
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

    if calibration_file is None:
        calibration_file = _cfg_calibration(cfg, "neutron_tof_calibration_path")
    if calibration_file is not None:
        dt = time_bins_s[1] - time_bins_s[0] if len(time_bins_s) > 1 else 1.0
        t_resp, r_resp = _load_calibration_curve(calibration_file, "tof")
        hist = _apply_instrument_response(hist, dt, t_resp, r_resp)
    return _apply(hist, response_fn, noise_fn)


def angular_neutron_spectrum(
    angles_deg: Sequence[float],
    base_yield: float,
    anisotropy: float = 0.0,
) -> List[float]:
    """Return a cosine-based angular neutron spectrum.

    This helper mirrors :func:`dpf2.diagnostics.neutron_spectra.angular_spectrum`
    but is provided here for lightweight synthetic diagnostics.
    """

    return [
        float(base_yield * (1.0 + anisotropy * math.cos(math.radians(a))))
        for a in angles_deg
    ]


__all__ = [
    "current_waveform",
    "voltage_waveform",
    "coupled_current_waveform",
    "coupled_voltage_waveform",
    "rogowski_signal",
    "bdot_signal",
    "sxr_diode_signal",
    "neutron_tof_signal",
    "angular_neutron_spectrum",
]
