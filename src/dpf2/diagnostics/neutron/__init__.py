"""Lightweight neutron diagnostic helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Dict, Any, Mapping
import json

# Re-export core synthesis utilities from :mod:`neutron_spectra`
from ..neutron_spectra import (
    Detector,
    DetectorLayout,
    synthetic_tof_spectrum,
    angular_spectrum,
    anisotropy_metric,
    forward_radial_backward_counts,
    anisotropy_ratios,
    load_detector_layout,
    time_resolved_spectra,
    directional_time_resolved_spectra,
)

from .angular_distribution import (
    per_angle_yield,
    forward_radial_backward_totals,
    directional_yield,
)


def load_response(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Load a neutron detector response description from *path*.

    Parameters
    ----------
    path:
        Location of the JSON configuration file.
    overrides:
        Optional mapping of values that override those read from *path*.  This
        provides a lightweight mechanism for user customisation of the detector
        description without having to modify the distributed parameter files.

    Notes
    -----
    The file is expected to contain JSON with optional keys ``gating``,
    ``dead_time`` and ``transfer_function``. ``gating`` should be a mapping with
    ``start`` and ``end`` times.  ``transfer_function`` is interpreted as a
    kernel for a simple discrete convolution applied after gating and dead
    time.
    """
    with open(Path(path), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if overrides:
        data.update(overrides)
    return data


def apply_response(
    times: Sequence[float],
    signal: Sequence[float],
    response: Dict[str, Any],
) -> List[float]:
    """Apply detector response effects to *signal* sampled at *times*.

    Gating removes samples outside the time window.  Dead time suppresses
    subsequent non-zero samples that occur within the specified interval after
    a non-zero sample.  Dispersion convolves the signal with the provided
    kernel.
    """
    if len(times) != len(signal):
        raise ValueError("times and signal must be the same length")

    vals = [float(v) for v in signal]
    t = [float(x) for x in times]

    gate = response.get("gating")
    if gate:
        start = float(gate.get("start", -float("inf")))
        end = float(gate.get("end", float("inf")))
        vals = [v if start <= ti <= end else 0.0 for ti, v in zip(t, vals)]

    dead_time = response.get("dead_time")
    if dead_time is not None:
        dt = float(dead_time)
        last = -float("inf")
        processed: List[float] = []
        for ti, v in zip(t, vals):
            if v != 0.0 and ti - last < dt:
                processed.append(0.0)
            else:
                processed.append(v)
                if v != 0.0:
                    last = ti
        vals = processed

    kernel = response.get("transfer_function") or response.get("dispersion")
    if kernel:
        kernel = [float(k) for k in kernel]
        conv = [0.0 for _ in vals]
        for i, v in enumerate(vals):
            for j, k in enumerate(kernel):
                idx = i + j
                if idx < len(conv):
                    conv[idx] += v * k
        vals = conv

    return vals


def anisotropy_report(
    layout: DetectorLayout,
    spectra: Dict[str, Sequence[float]],
) -> Dict[str, Any]:
    """Aggregate detector counts and compute anisotropy metrics.

    Parameters
    ----------
    layout:
        Detector arrangement used to determine detector orientation.
    spectra:
        Mapping of detector name to time-resolved counts.

    Returns
    -------
    dict
        A dictionary with ``"counts"`` holding forward/radial/backward totals,
        ``"ratios"`` with simple anisotropy ratios and ``"metric"`` providing a
        ``(max-min)/mean`` style anisotropy measure.
    """

    counts = forward_radial_backward_counts(layout, spectra)
    ratios = anisotropy_ratios(counts)
    metric = anisotropy_metric(list(counts.values()))
    return {"counts": counts, "ratios": ratios, "metric": metric}

__all__ = [
    "Detector",
    "DetectorLayout",
    "synthetic_tof_spectrum",
    "angular_spectrum",
    "anisotropy_metric",
    "forward_radial_backward_counts",
    "anisotropy_ratios",
    "load_detector_layout",
    "time_resolved_spectra",
    "directional_time_resolved_spectra",
    "per_angle_yield",
    "forward_radial_backward_totals",
    "directional_yield",
    "load_response",
    "apply_response",
    "anisotropy_report",
]

