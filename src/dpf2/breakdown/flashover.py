from __future__ import annotations

"""Flashover breakdown models using SEEA-based stochastic delay.

This module provides helpers for modeling flashover delays with
secondary electron emission avalanche (SEEA) concepts.  The approach is
intentionally lightweight and is meant to supply deterministic behaviour
for testing while exposing hooks for more detailed physics models.
"""

from dataclasses import dataclass
import math
import random
from typing import Sequence, Dict

from ..geometry import triple_junction_field


@dataclass
class FlashoverParameters:
    """Parameters controlling the stochastic delay model."""

    field_threshold: float
    sigma: float = 0.1
    conditioning: float = 0.0
    seed: int | None = None


def conditioning_curve(shot: int, alpha: float) -> float:
    """Return the conditioning multiplier for ``shot``.

    A simple exponential curve ``exp(-alpha * shot)`` is used.  ``shot``
    must be non-negative.  When ``alpha`` is zero the conditioning factor
    is unity for all shots.
    """

    if shot < 0:
        raise ValueError("shot must be non-negative")
    return math.exp(-alpha * shot)


def seea_stochastic_delay(
    field: float,
    params: FlashoverParameters,
    shot: int = 0,
) -> float:
    """Sample a stochastic flashover delay.

    The mean delay scales with the ratio ``field_threshold/field`` and a
    conditioning curve ``conditioning_curve(shot, params.conditioning)``.
    A log-normal distribution with standard deviation ``params.sigma`` is
    used to provide stochasticity.  ``params.seed`` can be specified to
    make the sampling reproducible for tests.
    """

    if field <= 0:
        raise ValueError("field must be positive")

    base = params.field_threshold / field
    conditioned = base * conditioning_curve(shot, params.conditioning)
    # Avoid log of zero/negative in degenerate cases
    mean = max(conditioned, 1e-12)
    mu = math.log(mean)
    rng = random.Random(params.seed)
    return rng.lognormvariate(mu, params.sigma)


def delay_series(
    field: float,
    params: FlashoverParameters,
    shots: int,
) -> Sequence[float]:
    """Generate a sequence of flashover delays over multiple shots."""

    delays = []
    for n in range(shots):
        # Vary the seed between shots to avoid identical samples when a
        # fixed seed is supplied
        p = FlashoverParameters(
            field_threshold=params.field_threshold,
            sigma=params.sigma,
            conditioning=params.conditioning,
            seed=None if params.seed is None else params.seed + n,
        )
        delays.append(seea_stochastic_delay(field, p, shot=n))
    return delays


def delay_statistics(delays: Sequence[float]) -> Dict[str, float]:
    """Return simple statistics for ``delays``.

    Statistics include the count, mean and standard deviation.
    """

    n = len(delays)
    if n == 0:
        return {"count": 0, "mean": 0.0, "stddev": 0.0}
    mean = sum(delays) / n
    var = sum((d - mean) ** 2 for d in delays) / n
    return {"count": n, "mean": mean, "stddev": var ** 0.5}


def holdoff_voltage(
    geometry: str,
    params: FlashoverParameters,
    shot: int = 0,
) -> float:
    """Sample a hold-off voltage for ``geometry``.

    The baseline hold-off is ``params.field_threshold`` scaled by the
    geometry-dependent triple-junction field factor.  Conditioning is modeled
    with :func:`conditioning_curve` such that the expected hold-off increases
    with shot count.  A log-normal distribution with ``params.sigma`` is used
    to introduce stochastic jitter.
    """

    if params.field_threshold <= 0:
        raise ValueError("field_threshold must be positive")

    tj_factor = triple_junction_field(geometry)
    base = params.field_threshold * tj_factor
    conditioned = base / conditioning_curve(shot, params.conditioning)
    mean = max(conditioned, 1e-12)
    mu = math.log(mean)
    rng = random.Random(params.seed)
    return rng.lognormvariate(mu, params.sigma)


def holdoff_series(
    geometry: str,
    params: FlashoverParameters,
    shots: int,
) -> Sequence[float]:
    """Generate a sequence of hold-off voltages over multiple shots."""

    values = []
    for n in range(shots):
        p = FlashoverParameters(
            field_threshold=params.field_threshold,
            sigma=params.sigma,
            conditioning=params.conditioning,
            seed=None if params.seed is None else params.seed + n,
        )
        values.append(holdoff_voltage(geometry, p, shot=n))
    return values


def jitter_statistics(jitters: Sequence[float]) -> Dict[str, float]:
    """Return statistics for jitter values."""

    return delay_statistics(jitters)

