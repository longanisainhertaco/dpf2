from __future__ import annotations

"""Radiation source models for the DPF solver."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..core_schema import RadiationModel, RadiationTransportModel
from .multigroup import MultiGroupDiffusion
from .power import bremsstrahlung_power, line_radiation_power
from .xray_emission_model import cr_line_emission

__all__ = [
    "RadiationBase",
    "BremsstrahlungModel",
    "MonteCarloRadiation",
    "create_radiation",
    "MultiGroupDiffusion",
    "bremsstrahlung_power",
    "line_radiation_power",
    "cr_line_emission",
]


class RadiationBase(Protocol):
    def loss(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return radiative energy loss rate [J/m^3/s]."""


@dataclass
class BremsstrahlungModel:
    coeff: float = 1e-5

    def loss(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        return self.coeff * rho**2 * np.sqrt(T)


@dataclass
class MonteCarloRadiation:
    base: RadiationBase
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def loss(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        mean = self.base.loss(rho, T)
        return self._rng.poisson(mean, size=mean.shape)


def create_radiation(
    model: RadiationModel,
    transport: RadiationTransportModel,
) -> RadiationBase | None:
    if model is RadiationModel.NONE:
        return None
    base = BremsstrahlungModel()
    if transport is RadiationTransportModel.MONTE_CARLO:
        return MonteCarloRadiation(base)
    return base
