from __future__ import annotations

"""Simplified radiation loss models for DPF simulations.

This module implements very approximate formulas for bremsstrahlung, line, and
recombination radiation losses.  The expressions are intended for order of
magnitude estimates and operate on cgs-like units:

* ``T_e`` [K] – electron temperature
* ``n_e``, ``n_i`` [m^-3] – electron and ion number densities
* ``Z_eff`` – effective ion charge state

References
----------
These simplified formulas are adapted from the NRL Plasma Formulary
:footcite:`Huba2016` and basic plasma radiation theory :footcite:`Bellan2008`.

The implementation purposefully favors clarity over accuracy; a more complete
model would tabulate emissivities from detailed collisional–radiative
calculations.
"""

from typing import Dict

import numpy as np

K_B = 1.380649e-23  # Boltzmann constant [J/K]


def bremsstrahlung_loss(T_e: np.ndarray, n_e: np.ndarray, n_i: np.ndarray, Z_eff: float = 1.0) -> np.ndarray:
    """Return bremsstrahlung power density ``P_brem`` [W/m^3].

    Uses the non–relativistic approximation :math:`P \approx 1.69\times10^{-32}
    Z_\mathrm{eff} n_e n_i \sqrt{T_e}` where ``T_e`` is in kelvin.
    """

    return 1.69e-32 * Z_eff * n_e * n_i * np.sqrt(T_e)


def recombination_loss(T_e: np.ndarray, n_e: np.ndarray, n_i: np.ndarray, Z_eff: float = 1.0) -> np.ndarray:
    """Return radiative recombination loss ``P_rec`` [W/m^3].

    Approximated by :math:`P \approx 1.7\times10^{-32} Z_\mathrm{eff} n_e n_i / \sqrt{T_e}`.
    """

    return 1.7e-32 * Z_eff * n_e * n_i / np.sqrt(T_e)


def line_loss(T_e: np.ndarray, n_e: np.ndarray, n_i: np.ndarray, Z_eff: float = 1.0) -> np.ndarray:
    """Very crude line radiation estimate ``P_line`` [W/m^3].

    Modeled as an exponential drop with temperature using an effective hydrogenic
    excitation energy (13.6 eV).
    """

    T_e_eV = T_e / 11604.525
    return 1e-31 * n_e * n_i * np.exp(-13.6 / np.maximum(T_e_eV, 1e-6))


def total_radiation_loss(T_e: np.ndarray, n_e: np.ndarray, n_i: np.ndarray, Z_eff: float = 1.0) -> Dict[str, np.ndarray]:
    """Return a dictionary containing individual and total radiation losses."""

    brem = bremsstrahlung_loss(T_e, n_e, n_i, Z_eff)
    line = line_loss(T_e, n_e, n_i, Z_eff)
    recomb = recombination_loss(T_e, n_e, n_i, Z_eff)
    total = brem + line + recomb
    return {"bremsstrahlung": brem, "line": line, "recombination": recomb, "total": total}

