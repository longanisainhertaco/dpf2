"""Calibration routines for inferring model parameters from diagnostics."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import math
import random

import numpy as np

Bounds = Dict[str, Tuple[float, float]]


def bayesian_calibration(
    model: Callable[[np.ndarray], np.ndarray],
    bounds: Bounds,
    data: np.ndarray,
    n_samples: int = 1000,
    proposal_scale: float = 0.1,
    sigma: float = 1.0,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Infer model parameters from experimental ``data`` using MCMC.

    This implements a simple Metropolis-Hastings sampler with uniform
    priors over ``bounds`` and a Gaussian likelihood with standard
    deviation ``sigma``.

    Parameters
    ----------
    model:
        Callable accepting an array of parameters and returning model
        predictions aligned with ``data``.
    bounds:
        Mapping of parameter names to ``(min, max)`` bounds.
    data:
        Experimental measurements to calibrate against.
    n_samples:
        Number of MCMC samples to draw.
    proposal_scale:
        Standard deviation of the Gaussian proposal distribution as a
        fraction of the parameter range.
    sigma:
        Standard deviation of the observational noise.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping each parameter name to an array of posterior
        samples.
    """

    rng = random.Random(seed)
    names = list(bounds)
    lower = [bounds[n][0] for n in names]
    upper = [bounds[n][1] for n in names]
    current = [(l + u) / 2.0 for l, u in zip(lower, upper)]

    def log_like(params: np.ndarray) -> float:
        pred = np.asarray(model(params))
        resid = (data - pred) / sigma
        return -0.5 * np.dot(resid, resid)

    samples: list[list[float]] = []
    current_logp = log_like(current)
    widths = [proposal_scale * (u - l) for l, u in zip(lower, upper)]

    for _ in range(n_samples):
        proposal = [rng.gauss(c, w) for c, w in zip(current, widths)]
        if all(l <= p <= u for p, l, u in zip(proposal, lower, upper)):
            logp = log_like(proposal)
            if math.log(rng.random()) < (logp - current_logp):
                current = proposal
                current_logp = logp
        samples.append(list(current))

    arr = np.array(samples)
    return {name: arr[:, idx] for idx, name in enumerate(names)}


def nested_calibration(
    model: Callable[[np.ndarray], np.ndarray],
    bounds: Bounds,
    data: np.ndarray,
    n_live: int = 50,
    n_iter: int = 500,
    sigma: float = 1.0,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Calibrate model parameters using a basic nested sampler.

    The implementation follows the classic nested sampling algorithm
    with uniform priors over ``bounds`` and a Gaussian likelihood with
    standard deviation ``sigma``.  It is intended for quick exploration
    against laboratory diagnostics where evaluating the forward model is
    expensive but derivatives are unavailable.

    Parameters
    ----------
    model:
        Callable accepting an array of parameters and returning model
        predictions aligned with ``data``.
    bounds:
        Mapping of parameter names to ``(min, max)`` bounds.
    data:
        Experimental measurements to calibrate against.
    n_live:
        Number of live points to maintain in the active set.
    n_iter:
        Number of nested sampling iterations to perform.
    sigma:
        Standard deviation of the observational noise.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping parameter names to posterior samples gathered
        during the nested sampling procedure.
    """

    rng = random.Random(seed)
    names = list(bounds)
    lower = [bounds[n][0] for n in names]
    upper = [bounds[n][1] for n in names]

    def log_like(params: np.ndarray) -> float:
        pred = np.asarray(model(params))
        resid = (data - pred) / sigma
        return -0.5 * np.dot(resid, resid)

    # Draw initial live points uniformly within bounds
    live = [
        [rng.uniform(l, u) for l, u in zip(lower, upper)]
        for _ in range(n_live)
    ]
    logl = [log_like(p) for p in live]

    collected: list[list[float]] = []
    for _ in range(n_iter):
        worst = min(range(len(logl)), key=lambda i: logl[i])
        collected.append(list(live[worst]))
        threshold = logl[worst]

        # Replace the worst point with a new sample above the threshold
        while True:
            cand = [rng.uniform(l, u) for l, u in zip(lower, upper)]
            cand_logl = log_like(cand)
            if cand_logl > threshold:
                live[worst] = cand
                logl[worst] = cand_logl
                break

    samples = np.array(collected)
    return {name: samples[:, idx] for idx, name in enumerate(names)}


__all__ = ["bayesian_calibration", "nested_calibration"]

