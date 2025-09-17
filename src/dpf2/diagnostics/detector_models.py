from __future__ import annotations

from typing import Sequence, List, Mapping, Any

import math
import random


def apply_irf(
    times: Sequence[float],
    signal: Sequence[float],
    irf: Mapping[str, Any],
) -> List[float]:
    """Apply an instrument response function to ``signal``.

    Parameters
    ----------
    times, signal:
        Sequences of equal length representing the sampled signal.
    irf:
        Mapping with optional keys ``transfer_function`` (sequence of kernel
        coefficients), ``gating`` (dict with ``start``/``end`` in seconds),
        ``dead_time`` (seconds) and ``noise`` (dict with ``stddev`` and optional
        ``seed``).
    """
    if len(times) != len(signal):
        raise ValueError("times and signal must be the same length")

    vals = [float(v) for v in signal]
    t = [float(x) for x in times]

    gate = irf.get("gating") if irf else None
    if gate:
        start = float(gate.get("start", -float("inf")))
        end = float(gate.get("end", float("inf")))
        vals = [v if start <= ti <= end else 0.0 for ti, v in zip(t, vals)]

    dead = irf.get("dead_time") if irf else None
    if dead is not None:
        dt = float(dead)
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

    kernel = None
    if irf:
        kernel = irf.get("transfer_function") or irf.get("dispersion")
    if kernel:
        ker = [float(k) for k in kernel]
        conv = [0.0 for _ in vals]
        for i, v in enumerate(vals):
            for j, k in enumerate(ker):
                idx = i + j
                if idx < len(conv):
                    conv[idx] += v * k
        vals = conv

    noise = irf.get("noise") if irf else None
    if noise:
        std = float(noise.get("stddev", 0.0))
        seed = noise.get("seed")
        rng = random.Random(seed)
        vals = [v + rng.gauss(0.0, std) for v in vals]

    return vals


def _solid_angle(area: float, distance: float) -> float:
    """Return the small-angle approximation for detector solid angle."""
    if area <= 0 or distance <= 0:
        raise ValueError("area and distance must be positive")
    return area / (distance**2)


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
    "apply_irf",
]
