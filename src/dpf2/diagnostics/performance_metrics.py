from __future__ import annotations

from typing import Dict


def compute_performance_metrics(
    yield_per_shot: float,
    *,
    rep_rate_hz: float,
    energy_out_j: float,
    energy_in_j: float,
    electrode_mass_g: float,
    erosion_per_shot_g: float,
) -> Dict[str, float]:
    """Compute basic performance KPIs for a DPF system.

    Parameters
    ----------
    yield_per_shot:
        Neutron yield produced per individual shot.
    rep_rate_hz:
        Firing repetition rate in Hertz.
    energy_out_j:
        Output energy per shot in joules.
    energy_in_j:
        Input energy per shot in joules.
    electrode_mass_g:
        Mass of the sacrificial electrode material in grams available before
        replacement is required.
    erosion_per_shot_g:
        Electrode mass lost per shot in grams.

    Returns
    -------
    dict
        Mapping containing ``yield_per_shot``, ``yield_per_hour``,
        ``wall_plug_efficiency`` and ``lifetime_hours``.
    """
    if rep_rate_hz < 0:
        raise ValueError("rep_rate_hz must be non-negative")

    yield_hour = yield_per_shot * rep_rate_hz * 3600.0

    wall = energy_out_j / energy_in_j if energy_in_j > 0 else 0.0

    if erosion_per_shot_g <= 0 or rep_rate_hz == 0:
        lifetime = float("inf")
    else:
        shots = electrode_mass_g / erosion_per_shot_g
        lifetime = shots / rep_rate_hz / 3600.0

    return {
        "yield_per_shot": float(yield_per_shot),
        "yield_per_hour": float(yield_hour),
        "wall_plug_efficiency": float(wall),
        "lifetime_hours": float(lifetime),
    }
