from __future__ import annotations

from typing import Sequence, Tuple, List


def compute_scope_trace(times: Sequence[float], values: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Baseline subtract a scope trace.

    Parameters
    ----------
    times:
        Sample times in seconds.
    values:
        Measured quantity at each time sample.

    Returns
    -------
    Tuple[List[float], List[float]]
        The original times and baseline-subtracted values.
    """
    times_list = [float(t) for t in times]
    values_list = [float(v) for v in values]
    if len(times_list) != len(values_list):
        raise ValueError("times and values must be the same length")
    if not times_list:
        return [], []
    mean = sum(values_list) / len(values_list)
    values_processed = [v - mean for v in values_list]
    return times_list, values_processed
