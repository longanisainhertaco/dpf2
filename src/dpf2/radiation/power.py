"""Elementary radiation power models.

This module provides lightweight helpers used in the unit tests to
approximate bremsstrahlung and line radiation losses.  The expressions are
not intended to be accurate; they merely capture the expected scaling with
electron/ion density and temperature.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bremsstrahlung_power", "line_radiation_power"]


def bremsstrahlung_power(
    ne: np.ndarray,
    ni: np.ndarray,
    Te: np.ndarray,
    *,
    Z_eff: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Return volumetric bremsstrahlung power.

    Parameters
    ----------
    ne, ni:
        Electron and ion number densities [m^-3].  Scalars or ``numpy`` arrays.
    Te:
        Electron temperature [eV or K].  Only relative scaling is used in the
        tests so the unit choice is immaterial.

    Z_eff:
        Effective charge state of the plasma.  A value greater than one
        mimics the enhancement of bremsstrahlung due to high-Z
        impurities.  The default of ``1.0`` preserves the previous
        behaviour used in the unit tests.

    Returns
    -------
    numpy.ndarray
        Power density [W/m^3] scaling as ``ne * ni * Z_eff * sqrt(Te)``.
    """

    ne = np.asarray(ne)
    ni = np.asarray(ni)
    Te = np.asarray(Te)
    Z_eff = np.asarray(Z_eff)
    coeff = 1.0
    return coeff * ne * ni * Z_eff * np.sqrt(Te)


def line_radiation_power(
    ne: np.ndarray,
    Te: np.ndarray,
    *,
    coeff: float | np.ndarray = 0.0,
    impurity_fraction: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Placeholder line-radiation loss model.

    The helper mirrors :func:`bremsstrahlung_power` but uses a configurable
    coefficient and assumes a density-squared dependence.  By default the
    coefficient is zero so that calling the function has no effect unless a
    user supplies a non-zero value.
    """

    ne = np.asarray(ne)
    Te = np.asarray(Te)
    coeff = np.asarray(coeff)
    impurity_fraction = np.asarray(impurity_fraction)
    return coeff * impurity_fraction * ne * ne * np.sqrt(Te)
