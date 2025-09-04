"""Yield calculation helpers for neutron diagnostics."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

from ..neutron_yield import (
    compute_beam_target_yield as _compute_beam_target_yield,
    compute_thermonuclear_yield as _compute_thermonuclear_yield,
    IonBeamEDF,
)
from .angular import _apply_optional_response


def thermonuclear_yield(
    reactivity: Sequence[float], ion_density: Sequence[float], dt: float
) -> float:
    """Return thermonuclear yield integrated over time.

    This is a thin wrapper over :func:`dpf2.diagnostics.neutron_yield.compute_thermonuclear_yield`
    provided here for completeness of the subpackage API.
    """

    return _compute_thermonuclear_yield(reactivity, ion_density, dt)


def beam_target_yield(
    ion_edf: IonBeamEDF,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    distance: float,
    time_bins: Sequence[float],
    response: dict | None = None,
) -> Tuple[list[float], list[list[float]]]:
    """Return ``dN/dΩ`` and ToF histograms for each detector angle.

    Parameters
    ----------
    ion_edf, cross_section, angles, distance, time_bins:
        Forwarded to :func:`dpf2.diagnostics.neutron_yield.compute_beam_target_yield`.
    response:
        Optional instrument response dictionary understood by
        :func:`dpf2.diagnostics.neutron.apply_response`.  When provided the
        response is applied to each time-of-flight histogram and integrated yield.

    Returns
    -------
    (yields, tofs): ``yields`` gives ``dN/dΩ`` for each angle and ``tofs`` the
    corresponding time-of-flight histograms.
    """

    yields, tofs = _compute_beam_target_yield(
        ion_edf, cross_section, angles, distance, time_bins
    )
    if response:
        processed_yields: list[float] = []
        processed_tofs: list[list[float]] = []
        mid = [0.5 * (time_bins[i] + time_bins[i + 1]) for i in range(len(time_bins) - 1)]
        for y, hist in zip(yields, tofs):
            hist = _apply_optional_response(mid, hist, response)
            y_resp = _apply_optional_response([0.0], [y], response)[0]
            processed_yields.append(y_resp)
            processed_tofs.append(hist)
        yields, tofs = processed_yields, processed_tofs
    return yields, tofs
