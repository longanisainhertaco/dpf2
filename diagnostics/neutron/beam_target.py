"""Beam-target neutron yield calculations."""

from __future__ import annotations

from typing import Callable, Sequence, List, Tuple

from dpf2.diagnostics.neutron_yield import compute_beam_target_yield as _compute_bt


def compute_yield(
    ion_edf: any,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    distance: float,
    time_bins: Sequence[float],
    response_fn: Callable[[float], float] | None = None,
    tof_hook: Callable[[Sequence[float], float], Sequence[float]] | None = None,
) -> Tuple[List[float], List[List[float]]]:
    """Return per-angle beam–target yield and time-of-flight histograms.

    Parameters
    ----------
    ion_edf:
        Provider implementing ``energy_distribution(angle_deg)`` which returns
        ``(energies, differential_flux)`` sequences.
    cross_section:
        Callable yielding reaction cross section for a given energy value.
    angles:
        Detector angles in degrees.
    distance:
        Source to detector distance in meters used for time-of-flight
        calculation.
    time_bins:
        Monotonic sequence of time bin edges in seconds for the histogram.
    response_fn:
        Optional callable applied to the integrated yield and histogram.
    tof_hook:
        Optional callable invoked for each histogram after response processing.
        The hook receives ``(hist, angle_deg)`` and must return an iterable of
        bin contents.  This provides a convenient extension point for later
        instrumentation.
    """

    yields, tofs = _compute_bt(
        ion_edf,
        cross_section,
        angles,
        distance,
        time_bins,
        response_fn=response_fn,
    )
    if tof_hook is not None:
        tofs = [list(tof_hook(hist, ang)) for hist, ang in zip(tofs, angles)]
    return yields, [ [float(v) for v in hist] for hist in tofs ]
