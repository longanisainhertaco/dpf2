from __future__ import annotations

"""Simple ionization model with equilibrium solver and collisional-radiative ODE."""

import math

# Physical constants
E_CHARGE = 1.602176634e-19  # Coulombs
K_B = 1.380649e-23  # Boltzmann constant (J/K)
IONIZATION_ENERGY_EV = 13.6  # Hydrogen ionization energy in eV


def _k_ion(T: float) -> float:
    """Electron impact ionization rate coefficient [m^3/s].

    A crude Arrhenius form is used merely for testing purposes.
    """

    return 5e-14 * math.exp(-IONIZATION_ENERGY_EV * E_CHARGE / (K_B * T))


def _k_rec(T: float) -> float:
    """Radiative recombination rate coefficient [m^3/s]."""

    # Simple power-law fit for demonstration purposes
    return 1e-16 * (T / 1.0e4) ** -0.7


def collisional_radiative_rhs(ne: float, n_total: float, T: float) -> float:
    """Time derivative of electron density for a minimal CR model."""

    ion = _k_ion(T) * ne * (n_total - ne)
    rec = _k_rec(T) * ne**2
    return ion - rec


def equilibrium_electron_density(
    n_total: float, T: float, *, tol: float = 1e-6, max_iter: int = 100
) -> float:
    """Solve for the equilibrium electron density using bisection.

    Parameters
    ----------
    n_total:
        Total particle density [m^-3].
    T:
        Electron temperature [K].
    tol:
        Absolute tolerance for the root-solver.
    max_iter:
        Maximum number of bisection iterations.
    """

    lo, hi = 0.0, float(n_total)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = collisional_radiative_rhs(mid, n_total, T)
        if abs(fmid) < tol:
            return mid
        if fmid > 0.0:
            lo = mid
        else:
            hi = mid
    return mid


def ionization_energy_sink(ne: float, n_total: float, T: float) -> float:
    """Energy loss rate due to ionization [J/m^3/s]."""

    rate = _k_ion(T) * ne * (n_total - ne)
    return rate * IONIZATION_ENERGY_EV * E_CHARGE


__all__ = [
    "equilibrium_electron_density",
    "collisional_radiative_rhs",
    "ionization_energy_sink",
]
