from __future__ import annotations

"""SEEA-based flashover model with conditioning history."""

from dataclasses import dataclass, field
from typing import List, Sequence

from ..breakdown.flashover import (
    FlashoverParameters,
    conditioning_curve,
    seea_stochastic_delay,
    holdoff_voltage,
)


@dataclass
class FlashoverModel:
    """Track flashover delays and hold-off with conditioning history.

    Parameters
    ----------
    geometry:
        Name of the electrode geometry preset used for triple-junction
        field enhancement.
    params:
        Configuration parameters controlling the stochastic delay model.
    """

    geometry: str
    params: FlashoverParameters
    shot: int = 0
    delay_history: List[float] = field(default_factory=list)
    holdoff_history: List[float] = field(default_factory=list)

    def sample_delay(self, field: float) -> float:
        """Sample a stochastic flashover delay for ``field``.

        The internal ``shot`` counter is incremented after sampling so that
        successive calls represent consecutive conditioning shots.
        """

        p = FlashoverParameters(
            field_threshold=self.params.field_threshold,
            sigma=self.params.sigma,
            conditioning=self.params.conditioning,
            seed=None if self.params.seed is None else self.params.seed + self.shot,
        )
        delay = seea_stochastic_delay(field, p, shot=self.shot)
        self.delay_history.append(delay)
        self.shot += 1
        return delay

    def delay_distribution(self, field: float, shots: int) -> Sequence[float]:
        """Return a series of stochastic delays over ``shots`` shots."""

        return [self.sample_delay(field) for _ in range(shots)]

    def sample_holdoff(self) -> float:
        """Sample a hold-off voltage for the configured geometry."""

        p = FlashoverParameters(
            field_threshold=self.params.field_threshold,
            sigma=self.params.sigma,
            conditioning=self.params.conditioning,
            seed=None if self.params.seed is None else self.params.seed + self.shot,
        )
        value = holdoff_voltage(self.geometry, p, shot=self.shot)
        self.holdoff_history.append(value)
        self.shot += 1
        return value

    def holdoff_series(self, shots: int) -> Sequence[float]:
        """Return a series of hold-off voltages over ``shots`` shots."""

        return [self.sample_holdoff() for _ in range(shots)]

    def conditioning_curve(self, shots: int) -> Sequence[float]:
        """Return the conditioning multipliers for the first ``shots`` shots."""

        return [conditioning_curve(n, self.params.conditioning) for n in range(shots)]


__all__ = ["FlashoverModel", "FlashoverParameters", "conditioning_curve"]
