from __future__ import annotations

"""Utilities for computing simple key performance indicators.

The production code only needs a very small subset of KPIs in order to feed
other parts of the UI.  The implementation here intentionally keeps the
interface extremely small and returns a dictionary so that the front-end can
consume it without having to understand any additional classes.
"""

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
    if energy_in_j < 0 or energy_out_j < 0:
        raise ValueError("energies must be non-negative")
    if electrode_mass_g < 0:
        raise ValueError("electrode_mass_g must be non-negative")
    if erosion_per_shot_g < 0:
        raise ValueError("erosion_per_shot_g must be non-negative")

    # How many neutrons are produced per hour.  ``rep_rate_hz`` is the firing
    # rate, so we convert to hours by multiplying by ``3600``.
    yield_hour = yield_per_shot * rep_rate_hz * 3600.0

    # The wall plug efficiency is defined as the ratio of output to input
    # energy.  If the input energy is zero the efficiency is defined to be zero
    # rather than raising an error in order to keep the API convenient for the
    # front end.
    wall = energy_out_j / energy_in_j if energy_in_j > 0 else 0.0

    # ``erosion_per_shot_g`` can be zero when erosion has not been measured.
    # In such cases the lifetime is effectively unbounded.
    if erosion_per_shot_g == 0 or rep_rate_hz == 0:
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
