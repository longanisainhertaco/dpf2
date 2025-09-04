"""Multi-objective optimization helpers."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import random

import numpy as np

Bounds = Dict[str, Tuple[float, float]]


def random_pareto_search(
    evaluate: Callable[[np.ndarray], Tuple[float, float]],
    bounds: Bounds,
    n_samples: int = 1000,
    seed: int | None = None,
) -> List[Dict[str, float]]:
    """Approximate the Pareto front for yield and spot size.

    This routine performs a random search over the parameter space defined by
    ``bounds``.  The ``evaluate`` callable should return ``(yield, spot_size)``
    for a given parameter vector.  Solutions with higher ``yield`` and lower
    ``spot_size`` are preferred.  The method returns the set of nondominated
    parameter dictionaries representing the estimated Pareto front.

    Parameters
    ----------
    evaluate:
        Callable accepting an array of parameters and returning
        ``(yield, spot_size)``.
    bounds:
        Mapping of parameter names to ``(min, max)`` bounds.
    n_samples:
        Number of random samples to draw.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    List[Dict[str, float]]
        Parameter dictionaries corresponding to the estimated Pareto front.
    """

    rng = random.Random(seed)
    names = list(bounds)
    lower = [bounds[n][0] for n in names]
    upper = [bounds[n][1] for n in names]

    params = [[rng.uniform(l, u) for l, u in zip(lower, upper)] for _ in range(n_samples)]
    scores = [evaluate(np.array(p)) for p in params]

    yields = [s[0] for s in scores]
    spots = [s[1] for s in scores]
    pareto_mask = [True] * n_samples

    for i in range(n_samples):
        if not pareto_mask[i]:
            continue
        for j in range(n_samples):
            if j == i:
                continue
            if (
                yields[j] >= yields[i]
                and spots[j] <= spots[i]
                and (yields[j] > yields[i] or spots[j] < spots[i])
            ):
                pareto_mask[i] = False
                break

    pareto_params = [p for p, keep in zip(params, pareto_mask) if keep]
    return [{name: float(p[idx]) for idx, name in enumerate(names)} for p in pareto_params]


__all__ = ["random_pareto_search"]

