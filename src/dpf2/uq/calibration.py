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


def emcee_calibrate_mass_current(
    current_sim: np.ndarray,
    current_data: np.ndarray,
    tof_sim: np.ndarray,
    tof_data: np.ndarray,
    bounds: Bounds | None = None,
    n_walkers: int = 32,
    n_steps: int = 1000,
    sigma_current: float = 1.0,
    sigma_tof: float = 1.0,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Estimate mass and current scaling factors using :mod:`emcee`.

    The function assumes ``current_sim`` and ``tof_sim`` represent baseline
    model predictions.  The unknown multiplicative ``mass_factor`` and
    ``current_factor`` scale these predictions to best match the measured
    ``current_data`` and ``tof_data``.

    Parameters
    ----------
    current_sim, current_data:
        Arrays of simulated and measured current waveforms.
    tof_sim, tof_data:
        Arrays of simulated and measured time-of-flight values.
    bounds:
        Optional mapping providing ``(min, max)`` pairs for ``mass_factor`` and
        ``current_factor``.  Defaults to ``(0.5, 1.5)`` for each if omitted.
    n_walkers, n_steps:
        Controls for the ensemble sampler.
    sigma_current, sigma_tof:
        Standard deviations of the observational noise for each dataset.
    seed:
        Optional seed for reproducibility.
    """

    try:  # pragma: no cover - dependency may be optional
        import emcee  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("emcee is required for emcee_calibrate_mass_current") from exc

    if bounds is None:
        bounds = {"mass_factor": (0.5, 1.5), "current_factor": (0.5, 1.5)}

    names = ["mass_factor", "current_factor"]
    lower = np.array([bounds[n][0] for n in names])
    upper = np.array([bounds[n][1] for n in names])

    def log_prob(theta: np.ndarray) -> float:
        if np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        mass_factor, current_factor = theta
        curr_pred = current_factor * np.asarray(current_sim)
        tof_pred = mass_factor * np.asarray(tof_sim)
        resid_current = (np.asarray(current_data) - curr_pred) / sigma_current
        resid_tof = (np.asarray(tof_data) - tof_pred) / sigma_tof
        return -0.5 * (
            np.dot(resid_current, resid_current) + np.dot(resid_tof, resid_tof)
        )

    rng = np.random.default_rng(seed)
    p0 = lower + (upper - lower) * rng.random((n_walkers, len(names)))
    sampler = emcee.EnsembleSampler(n_walkers, len(names), log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=n_steps // 2, flat=True)
    return {name: chain[:, idx] for idx, name in enumerate(names)}


__all__ = [
    "bayesian_calibration",
    "nested_calibration",
    "emcee_calibrate_mass_current",
]

