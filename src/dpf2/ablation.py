from __future__ import annotations

import math

__all__ = ["insulator_sleeve_area", "ablation_mass_energy_source"]


def insulator_sleeve_area(inner_radius: float, length: float) -> float:
    """Return inner surface area of cylindrical insulator sleeve.

    Parameters
    ----------
    inner_radius: float
        Inner radius in meters.
    length: float
        Sleeve length in meters.
    Returns
    -------
    float
        Surface area in square meters.
    """
    return 2.0 * math.pi * inner_radius * length


def ablation_mass_energy_source(ablation_rate: float, area: float, latent_heat: float) -> tuple[float, float]:
    """Compute mass and energy source due to insulator ablation.

    Parameters
    ----------
    ablation_rate: float
        Mass flux in kg/(m^2*s).
    area: float
        Ablating surface area in m^2.
    latent_heat: float
        Specific energy required for ablation in J/kg.

    Returns
    -------
    tuple
        (mass_source [kg/s], energy_source [W]).
    """
    mass_source = ablation_rate * area
    energy_source = mass_source * latent_heat
    return mass_source, energy_source
