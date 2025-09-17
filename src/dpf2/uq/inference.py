"""Advanced parameter inference routines leveraging external samplers."""
from __future__ import annotations

from typing import Callable, Dict, Tuple, Sequence

import math
import statistics
import numpy as np

Bounds = Dict[str, Tuple[float, float]]


def bayes_factor(log_evidence_a: float, log_evidence_b: float) -> float:
    """Return the Bayes factor between two models."""

    return math.exp(log_evidence_a - log_evidence_b)


def posterior_summary(
    samples: Dict[str, Sequence[float]],
    alpha: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Summarise posterior samples with mean, std and interval."""

    out: Dict[str, Dict[str, float]] = {}
    for name, vals in samples.items():
        seq = [float(v) for v in vals]
        if not seq:
            out[name] = {"mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0}
            continue
        lower_q = (1 - alpha) / 2
        upper_q = 1 - lower_q
        seq_sorted = sorted(seq)
        n = len(seq_sorted) - 1
        lower = seq_sorted[int(lower_q * n)]
        upper = seq_sorted[int(upper_q * n)]
        mean = statistics.fmean(seq)
        std = statistics.pstdev(seq) if len(seq) > 1 else 0.0
        out[name] = {"mean": mean, "std": std, "lower": float(lower), "upper": float(upper)}
    return out


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


def emcee_infer_waveform(
    time_sim: np.ndarray,
    current_sim: np.ndarray,
    time_data: np.ndarray,
    current_data: np.ndarray,
    bounds: Bounds | None = None,
    n_walkers: int = 32,
    n_steps: int = 1000,
    sigma: float = 1.0,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Infer mass and current scaling from waveform data using MCMC.

    The routine fits ``mass_factor`` and ``current_factor`` such that a
    simulated current waveform best matches experimental measurements.
    ``mass_factor`` scales the simulation time axis while
    ``current_factor`` scales the amplitude.
    """

    try:  # pragma: no cover - dependency may be optional
        import emcee  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("emcee is required for emcee_infer_waveform") from exc

    if bounds is None:
        bounds = {"mass_factor": (0.5, 1.5), "current_factor": (0.5, 1.5)}

    names = ["mass_factor", "current_factor"]
    lower = np.array([bounds[n][0] for n in names])
    upper = np.array([bounds[n][1] for n in names])

    time_sim = np.asarray(time_sim, dtype=float)
    current_sim = np.asarray(current_sim, dtype=float)
    time_data = np.asarray(time_data, dtype=float)
    current_data = np.asarray(current_data, dtype=float)

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        mass_factor, current_factor = theta
        scaled_time = mass_factor * time_sim
        sim_interp = np.interp(
            time_data, scaled_time, current_sim, left=0.0, right=0.0
        )
        pred = current_factor * sim_interp
        resid = (current_data - pred) / sigma
        return -0.5 * np.dot(resid, resid)

    rng = np.random.default_rng(seed)
    p0 = lower + (upper - lower) * rng.random((n_walkers, len(names)))
    sampler = emcee.EnsembleSampler(n_walkers, len(names), log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=n_steps // 2, flat=True)
    return {name: chain[:, idx] for idx, name in enumerate(names)}


def dynesty_infer_waveform(
    time_sim: np.ndarray,
    current_sim: np.ndarray,
    time_data: np.ndarray,
    current_data: np.ndarray,
    bounds: Bounds | None = None,
    n_live: int = 50,
    n_iter: int = 500,
    sigma: float = 1.0,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Infer waveform scaling factors using :mod:`dynesty` nested sampling."""

    try:  # pragma: no cover - dependency may be optional
        import dynesty  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("dynesty is required for dynesty_infer_waveform") from exc

    if bounds is None:
        bounds = {"mass_factor": (0.5, 1.5), "current_factor": (0.5, 1.5)}

    names = ["mass_factor", "current_factor"]
    lower = np.array([bounds[n][0] for n in names])
    upper = np.array([bounds[n][1] for n in names])

    time_sim = np.asarray(time_sim, dtype=float)
    current_sim = np.asarray(current_sim, dtype=float)
    time_data = np.asarray(time_data, dtype=float)
    current_data = np.asarray(current_data, dtype=float)

    def prior_transform(u: np.ndarray) -> np.ndarray:
        return lower + u * (upper - lower)

    def log_like(theta: np.ndarray) -> float:
        mass_factor, current_factor = theta
        scaled_time = mass_factor * time_sim
        sim_interp = np.interp(
            time_data, scaled_time, current_sim, left=0.0, right=0.0
        )
        pred = current_factor * sim_interp
        resid = (current_data - pred) / sigma
        return -0.5 * np.dot(resid, resid)

    sampler = dynesty.NestedSampler(
        log_like, prior_transform, len(names), nlive=n_live, seed=seed
    )
    sampler.run_nested(maxiter=n_iter, print_progress=False)
    res = sampler.results
    return {name: res.samples[:, idx] for idx, name in enumerate(names)}


__all__ = [
    "bayes_factor",
    "posterior_summary",
    "emcee_infer",
    "dynesty_infer",
    "emcee_infer_waveform",
    "dynesty_infer_waveform",
]
