from __future__ import annotations

"""Utilities for bundling multiple diagnostic outputs together.

This module collects the existing primitive diagnostic helpers into a single
entry point that produces a dictionary of neutron, X-ray, time-of-flight and
interferometry outputs.  It layers detector response modelling, simple cable
dispersion and benchmark overlays on top of the core physics calculations so
that acceptance tests can validate dual-channel yield separation and angular
distributions without re-implementing analysis logic.
"""

from typing import Any, Callable, Dict, Mapping, Sequence, List

import numpy as np

from .interferometry import interferometer_phase_shift
from .neutron_yield import IonBeamEDF, yield_components_with_anisotropy
from .xray_spectra import compute_xray_spectrum


def apply_cable_dispersion(signal: Sequence[float], tau: float, dt: float) -> List[float]:
    """Apply a simple first-order cable dispersion filter to ``signal``.

    Parameters
    ----------
    signal:
        Samples of the diagnostic waveform.
    tau:
        Characteristic time constant (seconds) of the dispersive link.  Values
        ``<= 0`` disable the filter and return a copy of the input ``signal``.
    dt:
        Sampling interval in seconds.

    Returns
    -------
    list of float
        The dispersed waveform.
    """

    if tau <= 0:
        return [float(v) for v in signal]
    alpha = dt / (tau + dt)
    out: List[float] = []
    prev = 0.0
    for val in signal:
        prev = prev + alpha * (float(val) - prev)
        out.append(prev)
    return out


def _within_pass_band_local(
    data: Sequence[float], reference: Sequence[float], band: float | Sequence[float]
) -> List[bool]:
    """Pass-band comparison compatible with lightweight numpy stubs."""

    data_arr = [float(v) for v in data]
    ref_arr = [float(v) for v in reference]
    if isinstance(band, Sequence) and not isinstance(band, (str, bytes)):
        band_arr = [float(b) for b in band]
        if len(band_arr) == 1:
            band_arr = band_arr * len(ref_arr)
    else:
        band_arr = [float(band) for _ in ref_arr]
    tol = [b * (abs(r) if r != 0 else 1.0) for b, r in zip(band_arr, ref_arr)]
    return [abs(d - r) <= t for d, r, t in zip(data_arr, ref_arr, tol)]


def assemble_diagnostic_outputs(
    ion_edf: IonBeamEDF | None = None,
    cross_section: Callable[[float], float] | None = None,
    angles: Sequence[float] | None = None,
    distance_m: float | None = None,
    time_bins: Sequence[float] | None = None,
    reactivity: Sequence[float] | None = None,
    ion_density: Sequence[float] | None = None,
    dt: float | None = None,
    xray_energies_keV: Sequence[float] | None = None,
    xray_intensities: Sequence[float] | None = None,
    xray_bins_keV: Sequence[float] | None = None,
    interferometer_density: Sequence[float] | None = None,
    interferometer_path: Sequence[float] | None = None,
    interferometer_wavelength: float | None = None,
    detector_response: Callable[[float], float] | None = None,
    tof_noise: Callable[[float], float] | None = None,
    cable_tau: float | None = None,
    benchmark_reference: Sequence[float] | None = None,
    benchmark_band: float | Sequence[float] = 0.1,
    interferometer_irf: Mapping[str, Any] | None = None,
    xray_response: Callable[[float], float] | None = None,
) -> Dict[str, Any]:
    """Generate a consolidated diagnostic output dictionary.

    Only the diagnostics with sufficient input data are included in the output
    so callers may supply a minimal subset when running lightweight checks.
    The returned mapping always exposes dual-channel neutron yield components
    and per-angle spectra when neutron inputs are provided, enabling benchmark
    overlays and acceptance tests to consume a consistent schema.
    """

    outputs: Dict[str, Any] = {}

    if (
        ion_edf is not None
        and cross_section is not None
        and angles is not None
        and distance_m is not None
        and time_bins is not None
        and reactivity is not None
        and ion_density is not None
        and dt is not None
    ):
        neutron = yield_components_with_anisotropy(
            ion_edf,
            cross_section,
            list(angles),
            float(distance_m),
            list(time_bins),
            list(reactivity),
            list(ion_density),
            float(dt),
        )
        total_bt = float(neutron["beam_target_total"])
        total_th = float(neutron["thermal_total"])
        grand_total = total_bt + total_th
        energy_partition = {
            "beam_target": total_bt,
            "thermonuclear": total_th,
            "beam_target_fraction": total_bt / grand_total if grand_total else 0.0,
            "thermonuclear_fraction": total_th / grand_total if grand_total else 0.0,
        }

        per_angle_total = neutron.get("angular_total", [])
        if detector_response is not None:
            per_angle_total = [float(detector_response(v)) for v in per_angle_total]
        per_angle_map = {float(a): per_angle_total[i] for i, a in enumerate(angles)}
        benchmark_overlay: Dict[str, Any] | None = None
        if benchmark_reference is not None:
            benchmark_overlay = {
                "reference": [float(v) for v in benchmark_reference],
                "within_band": _within_pass_band_local(
                    per_angle_total, benchmark_reference, benchmark_band
                ),
            }

        outputs["dual_channel_yield"] = {
            "beam_target": neutron.get("angular_beam_target", []),
            "thermonuclear": neutron.get("angular_thermal", []),
            "total": per_angle_total,
            "anisotropy": neutron.get("anisotropy", 0.0),
            "energy_partition": energy_partition,
            "tof_channels": neutron.get("tof_channels", {}),
        }
        outputs["angular_distribution"] = {
            "spectra": neutron.get("angular_spectra", {}),
            "anisotropy": neutron.get("anisotropy", 0.0),
            "per_angle_total": per_angle_map,
            "benchmark_overlay": benchmark_overlay,
        }

        tof_channels = neutron.get("tof_channels", {}).get("total", [])
        processed_tofs: List[List[float]] = []
        if time_bins:
            dt_bin = float(time_bins[1] - time_bins[0]) if len(time_bins) > 1 else 1.0
            for hist in tof_channels:
                vals = [float(v) for v in hist]
                if detector_response is not None:
                    vals = [float(detector_response(v)) for v in vals]
                if tof_noise is not None:
                    vals = [v + float(tof_noise(v)) for v in vals]
                if cable_tau is not None:
                    vals = apply_cable_dispersion(vals, float(cable_tau), dt_bin)
                processed_tofs.append(vals)
        aggregate_tof: List[float] = []
        if processed_tofs:
            aggregate_tof = [sum(bin_vals) for bin_vals in zip(*processed_tofs)]
        outputs["tof"] = {
            "time_bins": list(time_bins) if time_bins is not None else [],
            "raw": tof_channels,
            "processed": processed_tofs,
            "aggregate": aggregate_tof,
        }

    if (
        xray_energies_keV is not None
        and xray_intensities is not None
        and xray_bins_keV is not None
    ):
        centers, counts = compute_xray_spectrum(
            xray_energies_keV, xray_intensities, xray_bins_keV
        )
        if xray_response is not None:
            counts = [float(xray_response(c)) for c in counts]
        outputs["xray"] = {"energy_keV": centers, "spectrum": counts}

    if (
        interferometer_density is not None
        and interferometer_path is not None
        and interferometer_wavelength is not None
    ):
        phase = interferometer_phase_shift(
            interferometer_density,
            interferometer_path,
            float(interferometer_wavelength),
            irf=interferometer_irf,
        )
        if detector_response is not None:
            phase = float(detector_response(phase))
        outputs["interferometry"] = {"phase_shift_rad": phase}

    return outputs


__all__ = ["assemble_diagnostic_outputs", "apply_cable_dispersion"]
