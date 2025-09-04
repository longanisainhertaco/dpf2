"""Angular aggregation helpers for neutron diagnostics."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..neutron_spectra import (
    angular_spectrum,
    DetectorLayout,
    forward_radial_backward_counts,
)
from .base import apply_response as _apply_response


def _apply_optional_response(
    times: Sequence[float], signal: Sequence[float], response: Dict[str, float] | None
) -> List[float]:
    """Apply :func:`dpf2.diagnostics.neutron.apply_response` if *response* given."""

    if response:
        return _apply_response(times, signal, response)
    return [float(v) for v in signal]


def angular_distribution(
    angles: Sequence[float],
    base_yield: float,
    anisotropy: float = 0.0,
    response: Dict[str, float] | None = None,
) -> Tuple[List[float], Dict[str, float]]:
    """Return per-angle yields and grouped forward/radial/backward counts."""

    layout = DetectorLayout(angles=angles, distance_m=1.0)
    spectrum = angular_spectrum(layout.angles_deg(), base_yield, anisotropy)
    if response:
        spectrum = [
            _apply_optional_response([0.0], [s], response)[0] for s in spectrum
        ]
    spectra = {det.name: [s] for det, s in zip(layout.detectors, spectrum)}
    counts = forward_radial_backward_counts(layout, spectra)
    return spectrum, counts


__all__ = ["angular_distribution"]
