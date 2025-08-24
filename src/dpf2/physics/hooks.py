from __future__ import annotations

"""Auxiliary physics hooks for neutrals and wall ablation.

The functions in this module are intentionally small and serve primarily to
provide extension points for tests.  They are not meant to model detailed
plasma surface interaction physics but instead offer a consistent interface
for neutral species and ablating walls within the simplified MHD framework.
"""

from ..ablation import ablation_mass_energy_source


def neutral_density_source(rho_n: float, ionization_rate: float) -> float:
    """Return the depletion rate of neutral mass density.

    Parameters
    ----------
    rho_n:
        Neutral mass density.
    ionization_rate:
        Effective ionisation rate coefficient ``[1/s]``.
    """

    return -ionization_rate * rho_n


def wall_ablation_source(
    ablation_rate: float, area: float, latent_heat: float
) -> tuple[float, float]:
    """Mass and energy sources due to wall ablation.

    This is a thin wrapper around :func:`dpf2.ablation.ablation_mass_energy_source`.
    """

    return ablation_mass_energy_source(ablation_rate, area, latent_heat)


__all__ = ["neutral_density_source", "wall_ablation_source"]
