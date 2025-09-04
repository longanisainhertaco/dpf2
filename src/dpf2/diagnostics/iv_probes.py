from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Dict, Any, Mapping
import json
import math


def load_response(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Load an I-V probe response description from *path*.

    Parameters
    ----------
    path:
        Location of the JSON configuration file.
    overrides:
        Optional mapping of values that override those read from *path*.
    """
    with open(Path(path), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if overrides:
        data.update(overrides)
    return data


def _rlc_kernel(length: int, dt: float, R: float, L: float, C: float) -> List[float]:
    """Generate an impulse response for a simple series RLC circuit."""
    kernel: List[float] = []
    alpha = R / (2.0 * L) if L else 0.0
    w0 = 1.0 / math.sqrt(L * C) if L and C else 0.0
    for i in range(length):
        t = i * dt
        if alpha < w0:
            wd = math.sqrt(w0 * w0 - alpha * alpha)
            val = (1.0 / (L * wd)) * math.exp(-alpha * t) * math.sin(wd * t)
        elif alpha == w0:
            val = (t / L) * math.exp(-alpha * t) if L else 0.0
        else:
            sq = math.sqrt(alpha * alpha - w0 * w0)
            r1 = -alpha + sq
            r2 = -alpha - sq
            val = (1.0 / (L * (r2 - r1))) * (math.exp(r1 * t) - math.exp(r2 * t)) if L else 0.0
        kernel.append(val)
    return kernel


def apply_response(
    times: Sequence[float],
    signal: Sequence[float],
    response: Dict[str, Any],
) -> List[float]:
    """Apply the configured probe response effects to the input *signal*."""
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

    rlc = response.get("rlc")
    if rlc:
        R = float(rlc.get("R", 0.0))
        L = float(rlc.get("L", 0.0))
        C = float(rlc.get("C", 0.0))
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        kernel = _rlc_kernel(len(vals), dt, R, L, C)
        conv = [0.0 for _ in vals]
        for i, v in enumerate(vals):
            for j, k in enumerate(kernel):
                idx = i + j
                if idx < len(conv):
                    conv[idx] += v * k
        vals = conv

    return vals

__all__ = ["load_response", "apply_response"]
