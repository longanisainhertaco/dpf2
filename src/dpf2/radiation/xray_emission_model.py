from __future__ import annotations

"""Simple collisional–radiative model for Neon and Argon line emission.

This module provides a minimal implementation of a collisional–radiative (CR)
model for the strongest soft X-ray lines of Neon and Argon.  It is intended for
synthetic diagnostic work where only an order of magnitude estimate of the line
emissivity is required.  The model uses an extremely simplified CR formulation::

    \epsilon = A * n_e^2 * exp(-E_line / T_e)

where ``n_e`` is the electron density in cm^-3, ``T_e`` is the electron
temperature in eV and ``E_line`` is the line energy in eV.  ``A`` is an
empirical coefficient tuned to reproduce published SXR datasets.  The emitted
power is returned in arbitrary units proportional to photons/(cm^3 s).

The model currently tabulates a single dominant line for each species
(“Ne_X” and “Ar_Kalpha”) but can be easily extended by adding more
entries to :data:`NEON_LINES` and :data:`ARGON_LINES`.
"""

from dataclasses import dataclass
from math import exp
from typing import Dict, Literal


@dataclass(frozen=True)
class Line:
    """Atomic line description."""

    energy_eV: float
    coeff: float  # empirical scaling coefficient


# Line data -----------------------------------------------------------------
# Values loosely based on typical prominent SXR lines in Neon and Argon.  The
# coefficients were chosen to reproduce the reference values stored in
# ``Validation/sxr_reference.csv``.

NEON_LINES: Dict[str, Line] = {
    "Ne_X": Line(energy_eV=900.0, coeff=1e-30),
}

ARGON_LINES: Dict[str, Line] = {
    "Ar_Kalpha": Line(energy_eV=3000.0, coeff=5e-31),
}


def cr_line_emission(
    T_e_eV: float,
    n_e_cm3: float,
    species: Literal["Ne", "Ar"],
    *,
    impurity_fraction: float = 1.0,
) -> Dict[str, float]:
    """Return line emissivity for the requested species.

    Parameters
    ----------
    T_e_eV:
        Electron temperature in eV.
    n_e_cm3:
        Electron density in cm^-3.
    species:
        Ion species identifier (``"Ne"`` or ``"Ar"``).

    impurity_fraction:
        Multiplier representing the fractional abundance of the emitting
        species.  ``1.0`` corresponds to a pure plasma of the given
        species while smaller values scale the emissivity accordingly.

    Returns
    -------
    Dict[str, float]
        Mapping of line name to emissivity value.  The units are arbitrary but
        scale with ``n_e_cm3^2`` and have an exponential dependence on
        ``T_e_eV`` as described above.
    """

    if T_e_eV <= 0 or n_e_cm3 <= 0:
        raise ValueError("Temperature and density must be positive")
    if impurity_fraction < 0:
        raise ValueError("impurity_fraction must be non-negative")

    if species == "Ne":
        lines = NEON_LINES
    elif species == "Ar":
        lines = ARGON_LINES
    else:
        raise ValueError(f"Unsupported species '{species}'")

    emissivity: Dict[str, float] = {}
    n2 = (n_e_cm3**2) * impurity_fraction
    for name, line in lines.items():
        emissivity[name] = line.coeff * n2 * exp(-line.energy_eV / T_e_eV)
    return emissivity


__all__ = ["cr_line_emission", "NEON_LINES", "ARGON_LINES", "Line"]
