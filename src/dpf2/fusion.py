from __future__ import annotations

"""Fusion reaction models and neutron yield utilities."""

import numpy as np

__all__ = ["bosch_hale_dd", "dd_fusion_rates", "dd_channel_fractions"]


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
    try:
        return float(reactivity)
    except Exception:  # pragma: no cover - array-like input
        return reactivity


def dd_fusion_rates(
    T_keV: float,
    n_thermal: float,
    n_beam: float | None = None,
    beam_energy_keV: float | None = None,
) -> tuple[float, float]:
    """Return separate thermonuclear and beam–target D–D fusion rates.

    Parameters
    ----------
    T_keV:
        Thermal ion temperature.
    n_thermal:
        Thermal ion density (m^-3).
    n_beam, beam_energy_keV:
        Beam density and kinetic energy for beam–target reactions.  If either
        value is ``None`` the beam–target rate is zero.

    Returns
    -------
    tuple[float, float]
        Thermonuclear and beam–target reaction rates (m^-3 s^-1).
    """

    reactivity = bosch_hale_dd(T_keV)
    thermo = n_thermal ** 2 * reactivity

    beam = 0.0
    if n_beam and beam_energy_keV:
        # ``bosch_hale_dd`` provides <sigma v>; approximate cross section by
        # dividing by beam speed to obtain a crude beam–target rate.
        sigma_v = bosch_hale_dd(beam_energy_keV)
        beam = n_beam * n_thermal * sigma_v

    return float(thermo), float(beam)


def dd_channel_fractions(
    T_keV: float,
    n_thermal: float,
    n_beam: float | None = None,
    beam_energy_keV: float | None = None,
) -> dict[str, float]:
    """Return fractional contributions of thermonuclear and beam–target rates."""

    thermo, beam = dd_fusion_rates(T_keV, n_thermal, n_beam, beam_energy_keV)
    total = thermo + beam
    if total <= 0.0:
        return {"thermonuclear": 0.0, "beam_target": 0.0}
    return {
        "thermonuclear": thermo / total,
        "beam_target": beam / total,
    }

