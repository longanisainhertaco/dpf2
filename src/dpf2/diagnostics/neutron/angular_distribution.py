"""Angular distribution utilities for neutron diagnostics."""

from __future__ import annotations

from typing import Mapping, Sequence, Dict


def per_angle_yield(
    spectra: Mapping[float, Sequence[float]],
    responses: Mapping[float, Sequence[float]] | None = None,
) -> Dict[float, float]:
    """Integrate spectra at each angle to obtain per-angle yields.

    Parameters
    ----------
    spectra:
        Mapping of detector angle in degrees to a sequence of counts.
    responses:
        Optional mapping of detector angle to an instrument response curve.
        When provided, counts are weighted by the response curve before
        integration.

    Returns
    -------
    dict
        Mapping of angle to the integrated (optionally weighted) yield.
    """

    yields: Dict[float, float] = {}
    for angle, counts in spectra.items():
        vals = [float(c) for c in counts]
        if responses and angle in responses:
            resp = [float(r) for r in responses[angle]]
            if len(resp) != len(vals):
                raise ValueError("response curve length must match counts")
            yld = sum(v * r for v, r in zip(vals, resp))
        else:
            yld = sum(vals)
        yields[float(angle)] = float(yld)
    return yields


def forward_radial_backward_totals(yields: Mapping[float, float]) -> Dict[str, float]:
    """Aggregate per-angle yields into forward, radial, and backward totals."""

    totals = {"forward": 0.0, "radial": 0.0, "backward": 0.0}
    for angle, value in yields.items():
        ang = float(angle) % 360.0
        val = float(value)
        if ang <= 45.0 or ang >= 315.0:
            totals["forward"] += val
        elif 135.0 <= ang <= 225.0:
            totals["backward"] += val
        else:
            totals["radial"] += val
    return totals


def directional_yield(
    spectra: Mapping[float, Sequence[float]],
    responses: Mapping[float, Sequence[float]] | None = None,
) -> Dict[str, float]:
    """Convenience wrapper returning directional totals directly."""

    return forward_radial_backward_totals(per_angle_yield(spectra, responses))


__all__ = [
    "per_angle_yield",
    "forward_radial_backward_totals",
    "directional_yield",
]
