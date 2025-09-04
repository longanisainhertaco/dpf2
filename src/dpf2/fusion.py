from __future__ import annotations

"""Fusion reaction models and neutron yield utilities."""

import math
from typing import Sequence

import numpy as np

__all__ = [
    "bosch_hale_dd",
    "dd_fusion_rates",
    "dd_channel_fractions",
    "dd_beam_target_angular_spectrum",
    "dd_directional_yields",
]


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


def dd_beam_target_angular_spectrum(
    beam_energy_keV: float,
    n_beam: float,
    n_target: float,
    angles_deg: Sequence[float],
) -> np.ndarray:
    """Return simple beam–target yield per angle.

    The distribution assumes a ``0.5*(1+cos^2(theta))`` dependence which captures
    the forward/backward peaking of D–D beam–target fusion in a very crude
    fashion.  ``angles_deg`` should span ``-180`` to ``180`` degrees.
    """

    if n_beam <= 0 or n_target <= 0:
        return np.zeros(len(list(angles_deg)))
    # ``bosch_hale_dd`` returns <sigma v>; approximate cross section by dividing
    # by beam speed to provide a yield per solid angle.
    sigma_v = bosch_hale_dd(beam_energy_keV)
    e_j = beam_energy_keV * 1.0e3 * 1.602176634e-19
    m_d = 3.343583719e-27  # deuteron mass in kg
    v = math.sqrt(2.0 * e_j / m_d)
    sigma = sigma_v / v
    weights = [0.5 * (1.0 + math.cos(math.radians(a)) ** 2) for a in angles_deg]
    return np.asarray([n_beam * n_target * sigma * w for w in weights])


def dd_directional_yields(
    beam_energy_keV: float,
    n_beam: float,
    n_target: float,
    bins: int = 360,
) -> dict[str, float]:
    """Return forward, radial and backward yield components.

    The angular domain is partitioned into three sectors covering the forward
    (``|theta| < 30°``), radial (``30° ≤ |theta| ≤ 150°``) and backward
    (``|theta| > 150°``) directions.  The sum of the three components equals the
    total beam–target yield.
    """

    angles = [-180.0 + i * (360.0 / bins) for i in range(bins)]
    spectrum = dd_beam_target_angular_spectrum(
        beam_energy_keV, n_beam, n_target, angles
    )
    forward = radial = backward = 0.0
    for ang, val in zip(angles, spectrum):
        a = abs(float(ang))
        v = float(val)
        if a < 30.0:
            forward += v
        elif a > 150.0:
            backward += v
        else:
            radial += v
    return {"forward": forward, "radial": radial, "backward": backward}

