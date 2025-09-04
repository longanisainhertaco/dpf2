from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Dict, Any
import json


def load_response(path: str | Path) -> Dict[str, Any]:
    """Load an X-ray detector response description from *path*.

    The JSON format mirrors that used for neutron detectors.
    """
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_response(
    times: Sequence[float],
    signal: Sequence[float],
    response: Dict[str, Any],
) -> List[float]:
    """Apply detector response effects to *signal* sampled at *times*.

    This delegates to the same implementation as the neutron module.
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

    dispersion = response.get("dispersion")
    if dispersion:
        kernel = [float(k) for k in dispersion]
        conv = [0.0 for _ in vals]
        for i, v in enumerate(vals):
            for j, k in enumerate(kernel):
                idx = i + j
                if idx < len(conv):
                    conv[idx] += v * k
        vals = conv

    return vals

__all__ = ["load_response", "apply_response"]
