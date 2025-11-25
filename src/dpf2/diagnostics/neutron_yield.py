from __future__ import annotations

from pathlib import Path
from bisect import bisect_right
from typing import Callable, Protocol, Sequence, Tuple, Dict, List, Any

import h5py_stub as h5py  # type: ignore
import math
from ..neutron_yield_model import compute_directional_spectrum


def compute_neutron_yield(reaction_rate: Sequence[float], dt: float) -> float:
    """Compute total neutron yield from a reaction rate history.

    Parameters
    ----------
    reaction_rate:
        One dimensional array containing the neutron production rate in
        neutrons/second for each time sample.
    dt:
        Time step between samples in seconds.

    Returns
    -------
    float
        Integrated neutron yield over the provided history.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    total = 0.0
    for val in reaction_rate:
        total += float(val)
    return total * dt


class IonBeamEDF(Protocol):
    """Interface providing ion energy distributions by angle."""

    def energy_distribution(self, angle_deg: float) -> Tuple[Sequence[float], Sequence[float]]:
        """Return energies and differential flux for a detector angle."""


def compute_beam_target_yield(
    ion_edf: IonBeamEDF,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    distance: float,
    time_bins: Sequence[float],
    m_n: float = 1.674e-27,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> tuple[list[float], list[list[float]]]:
    """Integrate EDF×σ(E) and compute TOF histograms for each angle.

    Parameters
    ----------
    ion_edf:
        Provider of ion energy distributions.
    cross_section:
        Callable returning the reaction cross section for an energy value.
    angles:
        Detector angles in degrees for which to compute ``dN/dΩ``.
    distance:
        Distance from source to detector in meters for time-of-flight.
    time_bins:
        Monotonic sequence of time bin edges in seconds.
    m_n:
        Neutron mass used for flight time calculation in kg.
    response_fn, noise_fn:
        Optional callables applied to the integrated yield and TOF histogram.
        ``response_fn`` is evaluated first and ``noise_fn`` should return a
        noise contribution for the response-corrected value.

    Returns
    -------
    tuple of (yields, tofs)
        ``yields`` contains ``dN/dΩ`` for each requested angle and ``tofs``
        contains corresponding time-of-flight histograms.
    """

    if any(time_bins[i] >= time_bins[i + 1] for i in range(len(time_bins) - 1)):
        raise ValueError("time_bins must be monotonically increasing")

    yields: list[float] = []
    tofs: list[list[float]] = []
    for ang in angles:
        energies, dist = ion_edf.energy_distribution(float(ang))
        e = [float(v) for v in energies]
        f = [float(v) for v in dist]
        if len(e) != len(f):
            raise ValueError("energies and distribution must have the same length")
        if len(e) < 2:
            yields.append(0.0)
            tofs.append([0.0 for _ in range(len(time_bins) - 1)])
            continue
        integ = 0.0
        hist = [0.0 for _ in range(len(time_bins) - 1)]
        for i in range(len(e) - 1):
            e1, e2 = e[i], e[i + 1]
            f1, f2 = f[i], f[i + 1]
            s1, s2 = cross_section(e1), cross_section(e2)
            dE = e2 - e1
            contrib = 0.5 * (f1 * s1 + f2 * s2) * dE
            integ += contrib
            e_mid = (e1 + e2) / 2.0
            t = distance / math.sqrt(2.0 * e_mid / m_n)
            idx = bisect_right(time_bins, t) - 1
            if 0 <= idx < len(hist):
                hist[idx] += contrib
        if response_fn or noise_fn:
            hist_processed: list[float] = []
            for val in hist:
                if response_fn:
                    val = response_fn(val)
                if noise_fn:
                    val += noise_fn(val)
                hist_processed.append(val)
            hist = hist_processed
            yield_val = integ
            if response_fn:
                yield_val = response_fn(yield_val)
            if noise_fn:
                yield_val += noise_fn(yield_val)
        else:
            yield_val = integ
        yields.append(yield_val)
        tofs.append(hist)
    return yields, tofs


def _zero_lag_correlations(
    tof: Sequence[float], current: Sequence[float], voltage: Sequence[float]
) -> Dict[str, float]:
    """Return zero-lag correlations between ToF histogram and I/V waveforms."""

    if not (len(tof) == len(current) == len(voltage)):
        raise ValueError("ToF, current and voltage sequences must be the same length")

    def _corr(a: Sequence[float], b: Sequence[float]) -> float:
        mean_a = sum(a) / len(a) if a else 0.0
        mean_b = sum(b) / len(b) if b else 0.0
        num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        den = math.sqrt(
            sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
        )
        return num / den if den != 0 else 0.0

    power = [i * v for i, v in zip(current, voltage)]
    return {
        "current": _corr(tof, current),
        "voltage": _corr(tof, voltage),
        "power": _corr(tof, power),
    }


def _lagged_phase_metrics(
    tof: Sequence[float],
    signal: Sequence[float],
    dt: float,
) -> Dict[str, Any]:
    """Return lagged cross-correlation and best phase alignment."""

    if len(tof) != len(signal):
        raise ValueError("ToF and signal histories must have the same length")
    if dt <= 0:
        raise ValueError("dt must be positive for phase estimation")

    n = len(tof)
    lags: List[int] = []
    corr: List[float] = []
    mean_a = sum(tof) / n if n else 0.0
    mean_b = sum(signal) / n if n else 0.0
    for lag in range(-n + 1, n):
        val = 0.0
        for i in range(n):
            j = i - lag
            if 0 <= j < n:
                val += (tof[i] - mean_a) * (signal[j] - mean_b)
        lags.append(lag)
        corr.append(val)
    best_idx = max(range(len(corr)), key=lambda i: abs(corr[i])) if corr else 0
    best_lag = lags[best_idx] * dt if lags else 0.0
    return {"lags": [l * dt for l in lags], "correlation": corr, "best_lag_s": best_lag}


def compute_thermonuclear_yield(
    reactivity: Sequence[float], ion_density: Sequence[float], dt: float
) -> float:
    """Compute thermonuclear neutron yield from ion density and reactivity."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if len(reactivity) != len(ion_density):
        raise ValueError("reactivity and ion_density must be same length")
    rate = [r * n ** 2 for r, n in zip(reactivity, ion_density)]
    return compute_neutron_yield(rate, dt)


def yield_components_with_anisotropy(
    ion_edf: IonBeamEDF,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    distance: float,
    time_bins: Sequence[float],
    reactivity: Sequence[float],
    ion_density: Sequence[float],
    dt: float,
    current_trace: Sequence[float] | None = None,
    voltage_trace: Sequence[float] | None = None,
) -> Dict[str, List[float] | float]:
    """Return separated beam-target and thermal yields with angular spectra.

    The beam-target component is computed for each detector angle using
    :func:`compute_beam_target_yield` while the thermonuclear component is
    treated as isotropic.  The combined per-angle totals are used to compute an
    anisotropy factor ``(max-min)/mean``.  Both scalar totals and per-angle
    lists are exposed for downstream analysis along with channel-resolved time
    of flight (ToF) histograms and phase information derived from ``time_bins``.

    Parameters
    ----------
    ion_edf, cross_section, angles, distance, time_bins:
        Inputs forwarded to :func:`compute_beam_target_yield`.
    reactivity, ion_density, dt:
        Inputs forwarded to :func:`compute_thermonuclear_yield`.

    Returns
    -------
    dict
        Dictionary with keys ``"beam_target_total"`` and ``"thermal_total````
        for the integrated yields, ``"beam_target"`` and ``"angular_thermal"``
        for the corresponding per-angle distributions, ``"angular_total"`` for
        the sum of both components at each angle, ``"anisotropy"`` for the
        angular variation and ``"tof"`` containing time-of-flight histograms for
        each detector.
    """

    bt_yields, tofs = compute_beam_target_yield(
        ion_edf, cross_section, angles, distance, time_bins
    )
    th_total = compute_thermonuclear_yield(reactivity, ion_density, dt)
    # distribute thermal yield isotropically across angles
    th_per_angle = [th_total / float(len(angles)) for _ in angles] if angles else []
    bins = max(len(time_bins) - 1, 1)
    thermal_tof = [
        [th_total / max(len(angles), 1) / bins for _ in range(bins)] for _ in angles
    ]
    combined_tof = [
        [b + t for b, t in zip(bt_hist, th_hist)]
        for bt_hist, th_hist in zip(tofs, thermal_tof)
    ]
    total_per_angle = [b + t for b, t in zip(bt_yields, th_per_angle)]
    if total_per_angle:
        mean = sum(total_per_angle) / len(total_per_angle)
        if mean == 0.0:
            anisotropy = 0.0
        else:
            anisotropy = (max(total_per_angle) - min(total_per_angle)) / mean
    else:
        anisotropy = 0.0
    midpoints = [
        0.5 * (time_bins[i] + time_bins[i + 1]) for i in range(len(time_bins) - 1)
    ]
    window = time_bins[-1] - time_bins[0] if len(time_bins) > 1 else 0.0
    phase_fraction = [
        (t - time_bins[0]) / window if window > 0 else 0.0 for t in midpoints
    ]
    dt_hist = window / max(len(midpoints), 1) if midpoints else 0.0

    # Aggregate ToF signal for I–V phasing
    aggregate_tof = [sum(bin_vals) for bin_vals in zip(*combined_tof)] if combined_tof else []
    iv_phase: Dict[str, Any] | None = None
    if current_trace is not None and voltage_trace is not None and aggregate_tof:
        if len(current_trace) != len(aggregate_tof) or len(voltage_trace) != len(aggregate_tof):
            raise ValueError("current_trace and voltage_trace must match ToF histogram length")
        iv_phase = {
            "zero_lag": _zero_lag_correlations(aggregate_tof, current_trace, voltage_trace),
        }
        if dt_hist > 0:
            iv_phase["lagged"] = {
                "current": _lagged_phase_metrics(aggregate_tof, current_trace, dt_hist),
                "voltage": _lagged_phase_metrics(aggregate_tof, voltage_trace, dt_hist),
                "power": _lagged_phase_metrics(
                    aggregate_tof, [i * v for i, v in zip(current_trace, voltage_trace)], dt_hist
                ),
            }
        iv_phase["timebase_s"] = dt_hist
    return {
        "beam_target_total": sum(bt_yields),
        "thermal_total": th_total,
        # Per-angle distributions
        "beam_target": bt_yields,
        "angular_beam_target": bt_yields,
        "angular_thermal": th_per_angle,
        "angular_total": total_per_angle,
        "thermonuclear": th_total,
        "anisotropy": anisotropy,
        "tof": tofs,
        "tof_channels": {
            "beam_target": tofs,
            "thermonuclear": thermal_tof,
            "total": combined_tof,
        },
        "angular_spectra": {
            "beam_target": bt_yields,
            "thermonuclear": th_per_angle,
            "total": total_per_angle,
        },
        "tof_phase": {"time_midpoints": midpoints, "phase_fraction": phase_fraction},
        "iv_phase": iv_phase,
    }


def simulate_tof_detectors(
    ion_edf: IonBeamEDF,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    distance: float,
    time_bins: Sequence[float],
    detector_names: Sequence[str] | None = None,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> Dict[str, List[float]]:
    """Generate synthetic neutron time-of-flight detector histograms."""

    _, tofs = compute_beam_target_yield(
        ion_edf, cross_section, angles, distance, time_bins
    )
    dets: Dict[str, List[float]] = {}
    for i, hist in enumerate(tofs):
        processed = [response_fn(v) if response_fn else v for v in hist]
        if noise_fn:
            processed = [v + noise_fn(v) for v in processed]
        name = detector_names[i] if detector_names else f"detector_{i}"
        dets[name] = [float(v) for v in processed]
    return dets


def save_anisotropic_spectrum_hdf5(
    path: str | Path,
    energies: Sequence[float],
    angles: Sequence[float],
    spectrum: Sequence[Sequence[float]],
    detector_names: Sequence[str] | None = None,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
    openpmd: bool = False,
) -> None:
    """Save anisotropic neutron spectrum in an HDF5 file with per-detector datasets.

    Parameters
    ----------
    path:
        Output file location.
    energies, angles, spectrum:
        Spectral information for each detector angle.
    detector_names:
        Optional list naming each detector.
    response_fn, noise_fn:
        Optional callables applied to the spectral data.  ``response_fn`` is
        applied first to each value while ``noise_fn`` should return a noise
        contribution for the already response-corrected value which will be
        added to it.
    openpmd:
        If ``True`` the file will include a minimal openPMD structure with
        datasets stored under ``/data/0`` and standard attributes.
    """

    spec_arr = [[float(v) for v in row] for row in spectrum]
    if len(spec_arr) != len(angles) or any(len(row) != len(energies) for row in spec_arr):
        raise ValueError("spectrum shape must be (n_angles, n_energies)")
    if detector_names is not None and len(detector_names) != len(angles):
        raise ValueError("detector_names must match number of angles")

    with h5py.File(path, "w") as fh:
        base = fh
        if openpmd:
            fh.attrs.update(
                {
                    "openPMD": "1.1.0",
                    "basePath": "/data/%T/",
                    "iterationEncoding": "groupBased",
                    "iterationFormat": "%T",
                    "software": "dpf2",
                }
            )
            base = fh.require_group("data/0")
        e_ds = base.create_dataset("energy_MeV", data=[float(e) for e in energies])
        e_ds.data = list(e_ds.data)
        e_ds.attrs["unitSI"] = 1.0
        a_ds = base.create_dataset("angle_deg", data=[float(a) for a in angles])
        a_ds.data = list(a_ds.data)
        a_ds.attrs["unitSI"] = 1.0
        grp = base.require_group("detectors")
        for i, row in enumerate(spec_arr):
            processed = [response_fn(v) if response_fn else v for v in row]
            if noise_fn:
                processed = [v + noise_fn(v) for v in processed]
            name = detector_names[i] if detector_names else f"detector_{i}"
            ds = grp.create_dataset(name, data=processed)
            ds.data = list(processed)
            ds.attrs["unitSI"] = 1.0


def save_tof_hdf5(
    path: str | Path,
    time_bins: Sequence[float],
    detectors: Dict[str, Sequence[float]],
    openpmd: bool = False,
) -> None:
    """Export synthetic time-of-flight detector data to an HDF5 file."""

    with h5py.File(path, "w") as fh:
        base = fh
        if openpmd:
            fh.attrs.update(
                {
                    "openPMD": "1.1.0",
                    "basePath": "/data/%T/",
                    "iterationEncoding": "groupBased",
                    "iterationFormat": "%T",
                    "software": "dpf2",
                }
            )
            base = fh.require_group("data/0")
        t_ds = base.create_dataset("time_s", data=[float(t) for t in time_bins])
        t_ds.data = list(t_ds.data)
        t_ds.attrs["unitSI"] = 1.0
        grp = base.require_group("detectors")
        for name, hist in detectors.items():
            ds = grp.create_dataset(name, data=[float(v) for v in hist])
            ds.data = list(ds.data)
            ds.attrs["unitSI"] = 1.0


def tof_iv_cross_correlation(
    tof: Sequence[float],
    current: Sequence[float],
    voltage: Sequence[float],
) -> Dict[str, float]:
    """Return zero-lag correlation between TOF signal and I/V traces.

    In addition to individual current and voltage correlations the
    correlation with instantaneous electrical power (``I*V``) is also
    returned.  All inputs must be the same length and are treated as
    uniformly sampled in time.
    """

    if not (len(tof) == len(current) == len(voltage)):
        raise ValueError("signals must have the same length")

    def _corr(a: Sequence[float], b: Sequence[float]) -> float:
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        den = math.sqrt(
            sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
        )
        return num / den if den != 0 else 0.0

    power = [i * v for i, v in zip(current, voltage)]
    return {
        "current": _corr(tof, current),
        "voltage": _corr(tof, voltage),
        "power": _corr(tof, power),
    }


def ez_beam_correlation(
    ez: Sequence[float],
    ion_beam: Sequence[float],
    electron_beam: Sequence[float],
) -> Dict[str, float]:
    """Return zero-lag correlation between ``E_z`` and beam signals."""

    if not (len(ez) == len(ion_beam) == len(electron_beam)):
        raise ValueError("signals must have the same length")

    def _corr(a: Sequence[float], b: Sequence[float]) -> float:
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        den = math.sqrt(
            sum((x - mean_a) ** 2 for x in a)
            * sum((y - mean_b) ** 2 for y in b)
        )
        return num / den if den != 0 else 0.0

    return {
        "ion": _corr(ez, ion_beam),
        "electron": _corr(ez, electron_beam),
    }


def angular_yield_map(
    ion_edf: IonBeamEDF,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    energy_bins: Sequence[float],
) -> List[List[float]]:
    """Wrapper around :func:`compute_directional_spectrum` for diagnostics."""

    return compute_directional_spectrum(ion_edf, cross_section, angles, energy_bins)


def save_angular_yield_map_hdf5(
    path: str | Path,
    energy_bins: Sequence[float],
    angles: Sequence[float],
    spectrum: Sequence[Sequence[float]],
    detector_names: Sequence[str] | None = None,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
    openpmd: bool = False,
) -> None:
    """Export angular yield map to HDF5 using standard spectrum layout."""
    energies = [
        (energy_bins[i] + energy_bins[i + 1]) / 2.0
        for i in range(len(energy_bins) - 1)
    ]
    save_anisotropic_spectrum_hdf5(
        path,
        energies,
        angles,
        spectrum,
        detector_names=detector_names,
        response_fn=response_fn,
        noise_fn=noise_fn,
        openpmd=openpmd,
    )


__all__ = [
    "IonBeamEDF",
    "compute_neutron_yield",
    "compute_beam_target_yield",
    "compute_thermonuclear_yield",
    "yield_components_with_anisotropy",
    "save_anisotropic_spectrum_hdf5",
    "simulate_tof_detectors",
    "save_tof_hdf5",
    "tof_iv_cross_correlation",
    "ez_beam_correlation",
    "angular_yield_map",
    "save_angular_yield_map_hdf5",
]
