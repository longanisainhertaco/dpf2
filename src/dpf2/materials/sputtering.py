"""Sputtering models and impurity source terms.

This module provides minimal implementations of the Sigmund and
Yamamura sputtering yield formulas.  The expressions are simplified so
that the unit tests can exercise basic scaling behaviour without
requiring extensive atomic data tables.  The routines operate on
fundamental atomic properties (charge ``Z`` and mass in amu) and return
species yields and impurity source terms suitable for coupling with the
light–weight material damage model used in the test suite.

The Sigmund model describes the sputtering yield for normally incident
ions while the Yamamura model adds an empirical angular dependence.
Both models expose a similar interface and can be used to construct
impurity source terms from an incident ion flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, radians
from typing import Dict

__all__ = [
    "Species",
    "sigmund_yield",
    "yamamura_yield",
    "impurity_source_terms",
    "sputter_flux",
]


@dataclass(frozen=True)
class Species:
    """Basic atomic description used by the sputtering helpers."""

    name: str
    Z: int  # atomic number
    mass_u: float  # atomic mass in atomic mass units


# ----------------------------------------------------------------------------
# Yield models
# ----------------------------------------------------------------------------


def _threshold_energy(projectile: Species, target: Species, U0_eV: float) -> float:
    """Return the Sigmund threshold energy ``Eth``.

    The helper is primarily used to keep :func:`sigmund_yield` tidy and is
    exposed for testability.  ``U0_eV`` corresponds to the effective surface
    binding energy of the target.
    """

    mu = projectile.mass_u / target.mass_u
    return U0_eV * (1.0 + mu) ** 2 / (4.0 * mu)

def sigmund_yield(
    projectile: Species,
    target: Species,
    energy_eV: float,
    *,
    U0_eV: float = 4.0,
) -> float:
    """Return the Sigmund sputtering yield for normal incidence.

    Parameters
    ----------
    projectile, target:
        :class:`Species` instances describing the incoming ion and the
        target material.
    energy_eV:
        Incident ion energy in electron volts.
    U0_eV:
        Effective surface binding energy of the target in eV.  The
        default ``4.0`` eV roughly corresponds to typical metallic
        surfaces and keeps the test values simple.

    Notes
    -----
    The implementation follows the structure of Sigmund's theory but
    collapses the stopping cross section to a simple charge dependent
    factor.  The numerical coefficients are chosen so that the results
    are within the expected order of magnitude for common materials
    while remaining computationally lightweight.
    """

    if energy_eV <= 0 or U0_eV <= 0:
        return 0.0

    mu = projectile.mass_u / target.mass_u
    alpha = 0.08 + 0.164 * mu + 0.014 * mu * mu
    Sn = (projectile.Z * target.Z) / (
        projectile.Z ** (2.0 / 3.0) + target.Z ** (2.0 / 3.0)
    )

    # Threshold energy for sputtering [eV]
    Eth = _threshold_energy(projectile, target, U0_eV)
    if energy_eV < Eth:
        return 0.0

    return 0.042 * alpha * Sn * (1.0 - Eth / energy_eV)


def yamamura_yield(
    projectile: Species,
    target: Species,
    energy_eV: float,
    angle_deg: float,
    *,
    U0_eV: float = 4.0,
    f: float = 2.5,
) -> float:
    """Return the Yamamura sputtering yield for an arbitrary angle.

    The function first evaluates :func:`sigmund_yield` and then applies
    the Yamamura angular correction.  ``angle_deg`` is the angle between
    the incident direction and the surface normal.
    """

    Y0 = sigmund_yield(projectile, target, energy_eV, U0_eV=U0_eV)
    if Y0 <= 0:
        return 0.0

    theta = radians(angle_deg)
    c = cos(theta)
    if c <= 0.0:
        return 0.0

    return Y0 * (c ** -f) * exp(-f * (1.0 - c))


# ----------------------------------------------------------------------------
# Impurity source terms
# ----------------------------------------------------------------------------

def impurity_source_terms(
    ion_flux: float,
    yield_per_ion: float,
    species: Species,
) -> Dict[str, float]:
    """Return impurity source terms for a given incident flux.

    Parameters
    ----------
    ion_flux:
        Incoming ion flux [m^-2 s^-1].
    yield_per_ion:
        Sputtering yield for the incident species.
    species:
        Target material species being sputtered.

    Returns
    -------
    Dict[str, float]
        Mapping of species name to sputtered particle flux.  The flux is
        ``ion_flux * yield_per_ion`` with no additional scaling so that
        the values can be directly fed into simple impurity models.
    """

    if ion_flux <= 0 or yield_per_ion <= 0:
        return {species.name: 0.0}
    return {species.name: ion_flux * yield_per_ion}


def sputter_flux(
    projectile: Species,
    target: Species,
    ion_flux: float,
    energy_eV: float,
    angle_deg: float = 0.0,
    *,
    U0_eV: float = 4.0,
) -> Dict[str, float]:
    """Return impurity fluxes produced by a plasma hitting a surface.

    The routine combines the Yamamura angular yield model with
    :func:`impurity_source_terms` so that unit tests can exercise basic
    plasma–material interaction pathways without requiring a full PMI
    package.
    """

    Y = yamamura_yield(projectile, target, energy_eV, angle_deg, U0_eV=U0_eV)
    return impurity_source_terms(ion_flux, Y, target)
