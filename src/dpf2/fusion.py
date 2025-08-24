from __future__ import annotations

"""Fusion reaction models and neutron yield utilities."""

import numpy as np

__all__ = ["bosch_hale_dd"]


def bosch_hale_dd(T_keV: float | np.ndarray) -> float | np.ndarray:
    """Approximate D-D reactivity from Bosch-Hale parameterization.

    Parameters
    ----------
    T_keV : float or ndarray
        Ion temperature in keV.
    Returns
    -------
    float or ndarray
        Reactivity in m^3/s.
    """
    T_keV = np.asarray(T_keV)
    # Coefficients adapted from NRL formulary (approximate)
    A = 2.33e-14
    B = -19.94
    C = 0.0
    reactivity = A * T_keV ** (2.0 / 3.0) * np.exp(B / T_keV ** (1.0 / 3.0))
    return reactivity

