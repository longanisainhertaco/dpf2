"""Post-processing helpers for uncertainty quantification."""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import statistics
import random
from pathlib import Path

import numpy as np


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


def variance_decomposition(
    samples: Sequence[Sequence[float]],
    values: Sequence[float],
    names: Sequence[str],
) -> Dict[str, float]:
    """Return variance contributions for each parameter.

    The function builds upon :func:`sobol_indices` and multiplies the
    resulting first-order indices by the total variance of ``values``.  The
    output dictionary contains a ``"total_variance"`` entry in addition to the
    per-parameter contributions.
    """

    indices = sobol_indices(samples, values, names)
    var_y = statistics.pvariance(list(values)) if values else 0.0
    contrib = {name: indices[name] * var_y for name in names}
    contrib["total_variance"] = var_y
    return contrib


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


def propagate_yield_pinch(
    samples: Sequence[Sequence[float]] | Dict[str, Sequence[float]],
    model: Callable[[np.ndarray], tuple[float, float]],
    outdir: str | Path = "validation",
    alpha: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Propagate parameter samples to yield and pinch-time uncertainties.

    Parameters
    ----------
    samples:
        Either an iterable of parameter vectors or a mapping of parameter
        names to sequences of values representing posterior samples.
    model:
        Callable returning ``(yield, pinch_time)`` for a given parameter
        vector.
    outdir:
        Directory where summary plots will be written.  Created if it does
        not yet exist.
    alpha:
        Confidence level used when computing uncertainty bands.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping with ``"neutron_yield"`` and ``"pinch_time"`` entries each
        containing statistics from :func:`uncertainty_band`.
    """

    if isinstance(samples, dict):
        names = list(samples)
        rows = zip(*(samples[n] for n in names))
    else:
        rows = samples

    yields: list[float] = []
    pinches: list[float] = []
    for row in rows:
        yld, pinch = model(np.asarray(row, dtype=float))
        yields.append(float(yld))
        pinches.append(float(pinch))

    stats = {
        "neutron_yield": uncertainty_band(yields, alpha),
        "pinch_time": uncertainty_band(pinches, alpha),
    }

    try:  # pragma: no cover - plotting is optional
        import matplotlib.pyplot as plt  # type: ignore

        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.hist(yields, bins=30, color="C0", alpha=0.7)
        plt.xlabel("Neutron yield")
        plt.ylabel("Frequency")
        plt.title("Neutron yield distribution")
        plt.tight_layout()
        plt.savefig(out / "neutron_yield.png")
        plt.close()

        plt.figure()
        plt.hist(pinches, bins=30, color="C1", alpha=0.7)
        plt.xlabel("Pinch time")
        plt.ylabel("Frequency")
        plt.title("Pinch timing distribution")
        plt.tight_layout()
        plt.savefig(out / "pinch_time.png")
        plt.close()
    except Exception:
        pass

    return stats


def propagate_jitter_voltage_pressure(
    model: Callable[[Sequence[float]], tuple[float, float]],
    voltage: float,
    pressure: float,
    voltage_jitter_pct: float,
    pressure_jitter_pct: float,
    n_samples: int = 1000,
    alpha: float = 0.95,
    seed: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Propagate bank voltage and gas pressure jitter through ``model``.

    ``model`` is expected to accept a two-element array ``[voltage, pressure]``
    and return ``(neutron_yield, pinch_time)``.  Voltage and pressure are
    perturbed with independent Gaussian noise according to the supplied
    relative jitter percentages.
    """

    rng = random.Random(seed)
    v_std = abs(voltage) * voltage_jitter_pct
    p_std = abs(pressure) * pressure_jitter_pct
    voltages = [rng.gauss(voltage, v_std) for _ in range(n_samples)]
    pressures = [rng.gauss(pressure, p_std) for _ in range(n_samples)]

    yields: list[float] = []
    pinches: list[float] = []
    for v, p in zip(voltages, pressures):
        yld, pinch = model([v, p])
        yields.append(float(yld))
        pinches.append(float(pinch))

    return {
        "neutron_yield": uncertainty_band(yields, alpha),
        "pinch_time": uncertainty_band(pinches, alpha),
    }


__all__ = [
    "sobol_indices",
    "variance_decomposition",
    "uncertainty_band",
    "propagate_yield_pinch",
    "propagate_jitter_voltage_pressure",
]
