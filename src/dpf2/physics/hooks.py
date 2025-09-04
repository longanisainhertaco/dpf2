from __future__ import annotations

"""Auxiliary physics hooks for neutrals and wall ablation.

The functions in this module are intentionally small and serve primarily to
provide extension points for tests.  They are not meant to model detailed
plasma surface interaction physics but instead offer a consistent interface
for neutral species and ablating walls within the simplified MHD framework.
"""

from ..ablation import ablation_mass_energy_source


def neutral_density_source(
    rho_n: float,
    ionization_rate: float,
    *,
    t: float = 0.0,
    puff_start: float | None = None,
    puff_end: float | None = None,
    mass_flow_rate: float = 0.0,
    volume: float = 1.0,
) -> float:
    """Rate of change of neutral density.

    The source term combines ionisation losses with a simple
    representation of a timed gas puff.  When ``puff_start <= t <=
    puff_end`` a mass flow ``mass_flow_rate`` (kg/s) is injected through a
    nozzle and converted into a density source by dividing by the system
    volume.

    Parameters
    ----------
    rho_n:
        Neutral mass density.
    ionization_rate:
        Effective ionisation rate coefficient ``[1/s]``.
    t:
        Simulation time ``[s]``.
    puff_start, puff_end:
        Start and end times of the gas puff ``[s]``.
    mass_flow_rate:
        Mass flow through the puff nozzle ``[kg/s]``.
    volume:
        Volume used to convert mass flow to density ``[m^3]``.

    Returns
    -------
    float
        Time derivative of ``rho_n`` accounting for both sources and
        sinks.
    """

    injection = 0.0
    if (
        puff_start is not None
        and puff_end is not None
        and puff_start <= t <= puff_end
        and mass_flow_rate > 0.0
    ):
        injection = mass_flow_rate / volume
    return injection - ionization_rate * rho_n


def wall_ablation_source(
    ablation_rate: float, area: float, latent_heat: float
) -> tuple[float, float]:
    """Mass and energy sources due to wall ablation.

    Parameters
    ----------
    ablation_rate:
        Mass flux leaving the wall ``[kg/(m^2*s)]``.
    area:
        Ablating surface area ``[m^2]``.
    latent_heat:
        Specific energy required for ablation ``[J/kg]``.

    Returns
    -------
    tuple
        ``(mass_source, energy_source)`` with units ``kg/s`` and ``W``.

    Notes
    -----
    This is a thin wrapper around :func:`dpf2.ablation.ablation_mass_energy_source`.
    """

    return ablation_mass_energy_source(ablation_rate, area, latent_heat)


__all__ = ["neutral_density_source", "wall_ablation_source"]
