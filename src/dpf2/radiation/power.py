"""Elementary radiation power models.

This module provides lightweight helpers used in the unit tests to
approximate bremsstrahlung and line radiation losses.  The expressions are
not intended to be accurate; they merely capture the expected scaling with
electron/ion density and temperature.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bremsstrahlung_power", "line_radiation_power"]


def bremsstrahlung_power(ne: np.ndarray, ni: np.ndarray, Te: np.ndarray) -> np.ndarray:
    """Return volumetric bremsstrahlung power.

    Parameters
    ----------
    ne, ni:
        Electron and ion number densities [m^-3].  Scalars or ``numpy`` arrays.
    Te:
        Electron temperature [eV or K].  Only relative scaling is used in the
        tests so the unit choice is immaterial.

    Returns
    -------
    numpy.ndarray
        Power density [W/m^3] scaling as ``ne * ni * sqrt(Te)``.
    """

    ne = np.asarray(ne)
    ni = np.asarray(ni)
    Te = np.asarray(Te)
    coeff = 1.0
    return coeff * ne * ni * np.sqrt(Te)


def line_radiation_power(ne: np.ndarray, Te: np.ndarray, *, coeff: float = 0.0) -> np.ndarray:
    """Placeholder line-radiation loss model.

    The helper mirrors :func:`bremsstrahlung_power` but uses a configurable
    coefficient and assumes a density-squared dependence.  By default the
    coefficient is zero so that calling the function has no effect unless a
    user supplies a non-zero value.
    """

    ne = np.asarray(ne)
    Te = np.asarray(Te)
    return coeff * ne * ne * np.sqrt(Te)
