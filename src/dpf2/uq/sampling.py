"""Sampling schemes for uncertainty quantification."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import random

try:  # pragma: no cover - optional dependency
    from scipy.stats import qmc
except Exception:  # pragma: no cover - fallback when SciPy not installed
    qmc = None

Bounds = Dict[str, Tuple[float, float]]


def latin_hypercube(bounds: Bounds, n_samples: int, seed: int | None = None) -> np.ndarray:
    """Generate samples using Latin hypercube sampling.

    Parameters
    ----------
    bounds:
        Mapping of parameter names to ``(min, max)`` tuples.
    n_samples:
        Number of samples to generate.
    seed:
        Optional seed for reproducibility.

    Returns
    -------
    ``(n_samples, d)`` array of samples where ``d`` is the number of
    parameters. Columns follow the order of ``bounds``.
    """

    names = list(bounds)
    d = len(names)
    if qmc is not None:
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        sample = sampler.random(n_samples)
    else:  # Basic LHS implementation using Python's random
        rng = random.Random(seed)
        sample = [[0.0] * d for _ in range(n_samples)]
        for j, name in enumerate(names):
            a, b = bounds[name]
            intervals = [(i + rng.random()) / n_samples for i in range(n_samples)]
            rng.shuffle(intervals)
            for i, val in enumerate(intervals):
                sample[i][j] = a + val * (b - a)
        return np.array(sample)
    lower = np.array([bounds[k][0] for k in names])
    upper = np.array([bounds[k][1] for k in names])
    return lower + sample * (upper - lower)


def sobol_sample(bounds: Bounds, n_samples: int, seed: int | None = None) -> np.ndarray:
    """Generate Sobol sequence samples.

    Parameters
    ----------
    bounds:
        Mapping of parameter names to ``(min, max)`` tuples.
    n_samples:
        Number of samples to generate.
    seed:
        Optional seed for reproducibility.

    Returns
    -------
    ``(n_samples, d)`` array of samples in the order of ``bounds``.
    """

    names = list(bounds)
    d = len(names)
    if qmc is not None:
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(n_samples)))
        sample = sampler.random_base2(m)[:n_samples]
        lower = np.array([bounds[k][0] for k in names])
        upper = np.array([bounds[k][1] for k in names])
        return lower + sample * (upper - lower)
    else:  # Fall back to simple random sampling
        rng = random.Random(seed)
        sample = [[rng.random() for _ in range(d)] for _ in range(n_samples)]
        for i in range(n_samples):
            for j, name in enumerate(names):
                a, b = bounds[name]
                sample[i][j] = a + sample[i][j] * (b - a)
        return np.array(sample)
