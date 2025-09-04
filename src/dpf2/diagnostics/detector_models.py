from __future__ import annotations

from typing import Sequence, List

import math


def _solid_angle(area: float, distance: float) -> float:
    """Return the small-angle approximation for detector solid angle."""
    if area <= 0 or distance <= 0:
        raise ValueError("area and distance must be positive")
    return area / (distance ** 2)


def cr39_response(yields: Sequence[float], area: float, distance: float) -> List[float]:
    """Estimate track density for a CR-39/RCF detector."""
    sa = _solid_angle(area, distance)
    return [float(y) * sa for y in yields]


# RCF shares the same simple model as CR-39; provide a convenience alias
rcf_response = cr39_response


def time_gated_scintillator_response(
    hist: Sequence[float],
    time_bins: Sequence[float],
    gate_start: float,
    gate_end: float,
    area: float,
    distance: float,
) -> float:
    """Integrate a TOF histogram within a gating window and apply geometry."""
    if len(time_bins) != len(hist) + 1:
        raise ValueError("time_bins must bracket histogram bins")
    if gate_start > gate_end:
        raise ValueError("gate_start must be <= gate_end")
    sa = _solid_angle(area, distance)
    total = 0.0
    for i, val in enumerate(hist):
        t_mid = (time_bins[i] + time_bins[i + 1]) / 2.0
        if gate_start <= t_mid <= gate_end:
            total += float(val)
    return total * sa


__all__ = [
    "cr39_response",
    "rcf_response",
    "time_gated_scintillator_response",
]
