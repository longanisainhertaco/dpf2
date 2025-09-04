"""Post-processing helpers for uncertainty quantification."""
from __future__ import annotations

from typing import Dict, Sequence

import statistics


def _to_matrix(samples: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert ``samples`` to a list-of-lists without requiring ``numpy``."""

    if hasattr(samples, "__array__"):
        try:
            return samples.tolist()  # type: ignore[attr-defined]
        except Exception:
            pass
    return [list(row) for row in samples]


def sobol_indices(
    samples: Sequence[Sequence[float]],
    values: Sequence[float],
    names: Sequence[str],
) -> Dict[str, float]:
    """Estimate first-order Sobol indices from ``samples`` and ``values``.

    The implementation uses a squared correlation coefficient as a cheap
    approximation of the sensitivity indices and avoids heavy numerical
    dependencies so it can run with the lightweight test ``numpy`` stub.
    """

    matrix = _to_matrix(samples)
    y = list(values)
    if len(y) < 2:
        return {name: 0.0 for name in names}
    mean_y = statistics.fmean(y)
    var_y = statistics.pvariance(y)
    indices: Dict[str, float] = {}
    for idx, name in enumerate(names):
        x = [row[idx] for row in matrix]
        if len(x) < 2:
            indices[name] = 0.0
            continue
        mean_x = statistics.fmean(x)
        var_x = statistics.pvariance(x)
        if var_x == 0 or var_y == 0:
            indices[name] = 0.0
            continue
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)
        indices[name] = (cov ** 2) / (var_x * var_y)
    return indices


def uncertainty_band(values: Sequence[float], alpha: float = 0.95) -> Dict[str, float]:
    """Compute mean, standard deviation and a central interval for ``values``."""

    vals = list(values)
    if not vals:
        return {"mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    vals_sorted = sorted(vals)
    n = len(vals_sorted) - 1
    lower_idx = int((1 - alpha) / 2 * n)
    upper_idx = int((alpha + (1 - alpha) / 2) * n)
    lower = vals_sorted[lower_idx]
    upper = vals_sorted[upper_idx]
    return {"mean": mean, "std": std, "lower": lower, "upper": upper}


__all__ = ["sobol_indices", "uncertainty_band"]
