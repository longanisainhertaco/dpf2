"""Bayesian calibration routines for model parameter inference."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import math

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

    rng = np.random.default_rng(seed)
    names = list(bounds)
    lower = np.array([bounds[n][0] for n in names])
    upper = np.array([bounds[n][1] for n in names])
    current = (lower + upper) / 2.0

    def log_like(params: np.ndarray) -> float:
        pred = np.asarray(model(params))
        resid = (data - pred) / sigma
        return -0.5 * np.dot(resid, resid)

    samples = np.zeros((n_samples, len(names)))
    current_logp = log_like(current)
    widths = proposal_scale * (upper - lower)

    for i in range(n_samples):
        proposal = rng.normal(current, widths)
        if np.all((proposal >= lower) & (proposal <= upper)):
            logp = log_like(proposal)
            if math.log(rng.random()) < (logp - current_logp):
                current = proposal
                current_logp = logp
        samples[i] = current

    return {name: samples[:, idx] for idx, name in enumerate(names)}
