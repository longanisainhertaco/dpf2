from __future__ import annotations

"""Simple collisional-radiative models for ionisation.

The implementations here are intentionally lightweight and operate on
scalar NumPy arrays.  They serve as stand-ins for full FLYCHK/CRM models
and allow unit tests to exercise chemistry hooks in the solver.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..core_schema import IonizationModel
from .kinetics import RateTable, RateEquations
from .metadata import DatasetMetadata, load_adas_metadata, load_lxcat_metadata

__all__ = [
    "ChemistryModel",
    "SahaEquilibrium",
    "CollisionalRadiativeEquilibrium",
    "FlychkTable",
    "create_chemistry",
    "RateTable",
    "RateEquations",
    "DatasetMetadata",
    "load_adas_metadata",
    "load_lxcat_metadata",
]


class ChemistryModel(Protocol):
    """Return average ionisation state ``Zbar`` for density and temperature."""

    def ionization_state(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Average charge state."""


@dataclass
class SahaEquilibrium:
    """Toy Saha equilibrium model."""

    Z: float = 1.0

    def ionization_state(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.full(T.shape, 0.5 * self.Z)


@dataclass
class FlychkTable:
    """Interpolate a pre-computed FLYCHK table (T vs Zbar)."""

    table_path: Path

    def __post_init__(self) -> None:
        data = np.loadtxt(self.table_path, delimiter=",", skiprows=1)
        self._T = data[:, 0]
        self._Z = data[:, 1]

    def ionization_state(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.interp(T, self._T, self._Z, left=self._Z[0], right=self._Z[-1])


@dataclass
class CollisionalRadiativeEquilibrium:
    """Blend Saha equilibrium with a reduced collisional–radiative model."""

    rates: RateTable | None = None
    levels: int = 3
    iterations: int = 4
    dt: float = 1e-9

    def ionization_state(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:  # noqa: ARG002
        if self.rates is None:
            return SahaEquilibrium().ionization_state(rho, T)

        T_arr = np.asarray(T)
        zbar = np.zeros_like(T_arr, dtype=float)
        for idx, t in np.ndenumerate(T_arr):
            model = RateEquations(self.rates, levels=self.levels)
            n = [1.0] + [0.0] * (self.levels - 1)
            for _ in range(self.iterations):
                n = model.step(n, float(t), self.dt)
            zbar[idx] = model.mean_charge(n)
        return zbar


def create_chemistry(model: IonizationModel, *, table_path: Path | None = None) -> ChemistryModel:
    if model is IonizationModel.SAHA:
        return SahaEquilibrium()
    if model is IonizationModel.FLYCHK and table_path is not None:
        return FlychkTable(Path(table_path))
    if model is IonizationModel.CR:
        rates = RateTable.from_csv(table_path) if table_path is not None else None
        return CollisionalRadiativeEquilibrium(rates=rates)
    raise ValueError(f"Unsupported ionisation model: {model}")
