"""Geometry-aware plasma inductance utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:  # pragma: no cover - SciPy optional during tests
    from scipy.constants import mu_0
except Exception:  # pragma: no cover
    mu_0 = 4e-7 * np.pi

__all__ = [
    "CoaxialGeometry",
    "axial_inductance",
    "radial_inductance",
    "dynamic_inductance",
    "dynamic_inductance_with_derivatives",
]


@dataclass(frozen=True)
class CoaxialGeometry:
    """Electrode geometry description used by the inductance model."""

    anode_radius: float
    cathode_radius: float
    anode_length: float
    insulator_length: float = 0.0
    pinch_length: float | None = None
    end_correction: float = 0.0

    def __post_init__(self) -> None:
        if self.anode_radius <= 0.0:
            raise ValueError("anode_radius must be positive")
        if self.cathode_radius <= self.anode_radius:
            raise ValueError("cathode_radius must exceed anode_radius")
        if self.anode_length <= 0.0:
            raise ValueError("anode_length must be positive")

    @property
    def pinch_span(self) -> float:
        if self.pinch_length is not None:
            return self.pinch_length
        return 0.2 * self.anode_length

    @property
    def axial_gradient(self) -> float:
        return mu_0 / (2.0 * np.pi) * np.log(self.cathode_radius / self.anode_radius)

    @property
    def radial_gradient_scale(self) -> float:
        return mu_0 * self.pinch_span / (2.0 * np.pi)

    @property
    def insulator_inductance(self) -> float:
        if self.insulator_length <= 0.0:
            return 0.0
        return self.axial_gradient * self.insulator_length


def _clip(value: float | np.ndarray, lo: float, hi: float) -> float | np.ndarray:
    return np.clip(value, lo, hi)


def axial_inductance(z: float | np.ndarray, geom: CoaxialGeometry) -> float | np.ndarray:
    z_eff = _clip(z, 0.0, geom.anode_length)
    return geom.axial_gradient * z_eff


def radial_inductance(r: float | np.ndarray, geom: CoaxialGeometry) -> float | np.ndarray:
    r_min = 0.1 * geom.anode_radius
    r_eff = _clip(r, r_min, geom.cathode_radius)
    return geom.radial_gradient_scale * np.log(geom.cathode_radius / r_eff)


def dynamic_inductance(
    z: float | np.ndarray,
    r: float | np.ndarray,
    geom: CoaxialGeometry,
) -> float | np.ndarray:
    return (
        geom.end_correction
        + geom.insulator_inductance
        + axial_inductance(z, geom)
        + radial_inductance(r, geom)
    )


def dynamic_inductance_with_derivatives(
    z: float,
    r: float,
    geom: CoaxialGeometry,
) -> Tuple[float, float, float]:
    r_min = 0.1 * geom.anode_radius
    r_eff = float(_clip(r, r_min, geom.cathode_radius))
    L = dynamic_inductance(z, r_eff, geom)
    dL_dz = geom.axial_gradient if 0.0 < z < geom.anode_length else 0.0
    dL_dr = -geom.radial_gradient_scale / r_eff
    return L, dL_dz, dL_dr

