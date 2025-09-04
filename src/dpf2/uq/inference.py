"""Advanced parameter inference routines leveraging external samplers."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

Bounds = Dict[str, Tuple[float, float]]


def emcee_infer(
    model: Callable[[np.ndarray], np.ndarray],
    bounds: Bounds,
    data: np.ndarray,
    n_walkers: int = 32,
    n_steps: int = 1000,
    sigma: float = 1.0,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Infer parameters using the :mod:`emcee` ensemble sampler.

    The sampler uses uniform priors within ``bounds`` and a Gaussian
    likelihood with standard deviation ``sigma``.

    Parameters
    ----------
    model:
        Callable accepting an array of parameters and returning model
        predictions aligned with ``data``.
    bounds:
        Mapping of parameter names to ``(min, max)`` tuples.
    data:
        Experimental measurements to calibrate against.
    n_walkers:
        Number of walkers in the ensemble.
    n_steps:
        Total number of MCMC steps to draw.
    sigma:
        Standard deviation of the observational noise.
    seed:
        Optional seed for reproducibility.
    """

    try:  # pragma: no cover - dependency not installed in some environments
        import emcee  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("emcee is required for emcee_infer") from exc

    rng = np.random.default_rng(seed)
    names = list(bounds)
    lower = np.array([bounds[n][0] for n in names])
    upper = np.array([bounds[n][1] for n in names])
    ndim = len(names)

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        pred = np.asarray(model(theta))
        resid = (data - pred) / sigma
        return -0.5 * np.dot(resid, resid)

    p0 = lower + (upper - lower) * rng.random((n_walkers, ndim))
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=n_steps // 2, flat=True)
    return {name: chain[:, idx] for idx, name in enumerate(names)}


def dynesty_infer(
    model: Callable[[np.ndarray], np.ndarray],
    bounds: Bounds,
    data: np.ndarray,
    n_live: int = 50,
    n_iter: int = 500,
    sigma: float = 1.0,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Infer parameters using the :mod:`dynesty` nested sampler.

    Parameters mirror :func:`emcee_infer` but use a nested sampling
    approach suitable for multi-modal posteriors.
    """

    try:  # pragma: no cover - dependency not installed in some environments
        import dynesty  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("dynesty is required for dynesty_infer") from exc

    names = list(bounds)
    lower = np.array([bounds[n][0] for n in names])
    upper = np.array([bounds[n][1] for n in names])
    ndim = len(names)

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return lower + u * (upper - lower)

    def log_like(theta: np.ndarray) -> float:
        pred = np.asarray(model(theta))
        resid = (data - pred) / sigma
        return -0.5 * np.dot(resid, resid)

    sampler = dynesty.NestedSampler(
        log_like, prior_transform, ndim, nlive=n_live, seed=seed
    )
    sampler.run_nested(maxiter=n_iter, print_progress=False)
    res = sampler.results
    return {name: res.samples[:, idx] for idx, name in enumerate(names)}


__all__ = ["emcee_infer", "dynesty_infer"]
