"""Thermonuclear neutron yield calculations."""

from __future__ import annotations

from typing import Callable, Sequence, List, Tuple

from dpf2.diagnostics.neutron_yield import compute_thermonuclear_yield


def compute_yield(
    reactivity: Sequence[float],
    ion_density: Sequence[float],
    dt: float,
    angles: Sequence[float] | None = None,
    response_fn: Callable[[float], float] | None = None,
    tof_hook: Callable[[float, Sequence[float]], Sequence[float]] | None = None,
    time_bins: Sequence[float] | None = None,
) -> Tuple[List[float], List[List[float]]]:
    """Return per-angle thermonuclear yield and optional TOF histograms.

    Parameters
    ----------
    reactivity, ion_density, dt:
        Inputs forwarded to :func:`dpf2.diagnostics.neutron_yield.compute_thermonuclear_yield`.
    angles:
        Sequence of detector angles in degrees.  The thermonuclear yield is
        distributed isotropically across the provided angles.
    response_fn:
        Optional callable applied to the per-angle yield values.
    tof_hook:
        Optional callable used to generate a time-of-flight histogram for each
        angle.  The hook is invoked with the response-corrected per-angle yield
        and the ``time_bins`` sequence and must return an iterable of bin
        contents.
    time_bins:
        Monotonic sequence of time bin edges in seconds.  Required when
        ``tof_hook`` is provided.  If supplied without ``tof_hook`` empty
        histograms are returned, providing a convenient hook for later
        instrumentation.
    """

    total = compute_thermonuclear_yield(reactivity, ion_density, dt)
    n = len(angles) if angles else 0
    per_angle = [total / n for _ in range(n)] if n else []
    if response_fn:
        per_angle = [response_fn(v) for v in per_angle]
    tofs: List[List[float]] = []
    if time_bins is not None:
        for val in per_angle:
            hist = (
                list(tof_hook(val, time_bins))
                if tof_hook is not None
                else [0.0 for _ in range(len(time_bins) - 1)]
            )
            tofs.append([float(v) for v in hist])
    return per_angle, tofs
