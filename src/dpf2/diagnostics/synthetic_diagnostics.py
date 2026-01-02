"""Composite synthetic diagnostics with detector response support."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, List

from .detector_models import apply_irf
from .interferometry import interferometer_phase_shift
from .neutron_spectra import correlate_tof_peaks_with_circuit_iv
from .neutron_yield import IonBeamEDF, yield_components_with_anisotropy
from .xray_spectra import compute_xray_spectrum


def synthetic_neutron_diagnostics(
    ion_edf: IonBeamEDF,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    distance_m: float,
    time_bins: Sequence[float],
    reactivity: Sequence[float],
    ion_density: Sequence[float],
    dt: float,
    *,
    detector_irf: Mapping[str, Any] | None = None,
    current_trace: Sequence[float] | None = None,
    voltage_trace: Sequence[float] | None = None,
    align_to_iv: bool = False,
) -> Dict[str, Any]:
    """Return beam-target and thermonuclear channels with detector response."""

    hist_len = max(len(time_bins) - 1, 0)
    traces_match = (
        current_trace is not None
        and voltage_trace is not None
        and len(current_trace) == hist_len
        and len(voltage_trace) == hist_len
    )
    dual = yield_components_with_anisotropy(
        ion_edf,
        cross_section,
        angles,
        distance_m,
        time_bins,
        reactivity,
        ion_density,
        dt,
        current_trace=list(current_trace) if traces_match else None,
        voltage_trace=list(voltage_trace) if traces_match else None,
    )
    processed_tofs = []
    mid = [0.5 * (time_bins[i] + time_bins[i + 1]) for i in range(len(time_bins) - 1)]
    if detector_irf:
        for hist in dual["tof_channels"]["total"]:
            processed_tofs.append(apply_irf(mid, hist, detector_irf))
    else:
        processed_tofs = [list(hist) for hist in dual["tof_channels"]["total"]]

    aggregate = [sum(vals) for vals in zip(*processed_tofs)] if processed_tofs else []
    alignment: Dict[str, Any] | None = None
    if align_to_iv and aggregate and current_trace is not None and voltage_trace is not None:
        circuit_time = [i * dt for i in range(len(current_trace))]
        peaks, lags, corr, max_lag = correlate_tof_peaks_with_circuit_iv(
            time_bins, aggregate, circuit_time, current_trace, voltage_trace
        )
        dt_bin = time_bins[1] - time_bins[0] if len(time_bins) > 1 else 0.0
        shift_bins = int(round(max_lag / dt_bin)) if dt_bin else 0
        if shift_bins != 0:
            aggregate = _shift_histogram(aggregate, shift_bins)
            processed_tofs = [_shift_histogram(hist, shift_bins) for hist in processed_tofs]
        alignment = {"peaks": peaks, "lags": lags, "correlation": corr, "applied_shift_bins": shift_bins}

    return {
        "channels": {
            "beam_target": dual["angular_beam_target"],
            "thermonuclear": dual["angular_thermal"],
            "total": [b + t for b, t in zip(dual["angular_beam_target"], dual["angular_thermal"])],
        },
        "anisotropy": dual["anisotropy"],
        "tof": {"raw": dual["tof_channels"]["total"], "processed": processed_tofs, "aggregate": aggregate},
        "angular_spectra": dual["angular_spectra"],
        "alignment": alignment,
    }


def _shift_histogram(values: Sequence[float], bins: int) -> List[float]:
    """Shift a histogram by ``bins`` entries, zero padding the exposed edge."""

    vals = list(values)
    if bins == 0:
        return vals
    if bins > 0:
        return vals[bins:] + [0.0] * bins
    return [0.0] * (-bins) + vals[:bins]


def synthetic_xray_diagnostics(
    energies_keV: Sequence[float],
    intensities: Sequence[float],
    bins_keV: Sequence[float],
    *,
    detector_irf: Mapping[str, Any] | None = None,
) -> Tuple[List[float], List[float]]:
    """Compute an X-ray spectrum with optional detector response."""

    centers, spec = compute_xray_spectrum(energies_keV, intensities, bins_keV)
    if detector_irf:
        times = list(range(len(spec)))
        spec = apply_irf(times, spec, detector_irf)
    return centers, spec


def synthetic_interferometer_diagnostics(
    electron_density: Sequence[float],
    path_lengths: Sequence[float],
    wavelength: float,
    *,
    detector_irf: Mapping[str, Any] | None = None,
    response_fn: callable | None = None,
    noise_fn: callable | None = None,
) -> float:
    """Return a synthetic interferometer phase shift with detector response."""

    phase = interferometer_phase_shift(
        electron_density,
        path_lengths,
        wavelength,
        response_fn=response_fn,
        noise_fn=noise_fn,
        irf=detector_irf,
    )
    return phase


__all__ = [
    "synthetic_neutron_diagnostics",
    "synthetic_xray_diagnostics",
    "synthetic_interferometer_diagnostics",
]
