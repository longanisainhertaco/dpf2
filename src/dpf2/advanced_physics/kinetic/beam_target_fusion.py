"""Beam-target fusion yield model for D-D and D-T reactions.

This module provides cross-section parameterizations and yield
calculations for beam-target fusion in dense plasma focus devices,
where energetic ions from the pinch interact with thermal target
ions to produce fusion reactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

try:
    from scipy.constants import e as q_e, m_p
except ImportError:
    q_e = 1.602176634e-19
    m_p = 1.67262192369e-27

__all__ = [
    "dd_fusion_cross_section",
    "dt_fusion_cross_section",
    "beam_target_yield",
    "BeamTargetModel",
    "bosch_hale_reactivity",
]

# Atomic mass unit
AMU = 1.66053906660e-27

# Deuteron and triton masses
M_D = 2.014 * AMU
M_T = 3.016 * AMU


def dd_fusion_cross_section(E_keV: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute D-D fusion cross section using Bosch-Hale parameterization.

    This function returns the total D-D cross section (sum of D(d,n)He3
    and D(d,p)T branches).

    Parameters
    ----------
    E_keV : float or ndarray
        Center-of-mass energy [keV].

    Returns
    -------
    float or ndarray
        Cross section [m^2].
    """
    E_keV = np.asarray(E_keV, dtype=float)

    # Bosch-Hale coefficients for D(d,n)He3 branch
    B_G = 31.3970  # Gamow constant
    
    E_safe = np.maximum(E_keV, 1.0)  # Minimum 1 keV for stability

    # S-factor parameterization (keV-barns)
    # Using simplified form valid for 1-1000 keV
    S0 = 52.0  # S(0) in keV-barns
    S1 = 0.0  # First derivative term
    
    # Astrophysical S-factor
    S_factor = S0 * (1.0 + S1 * E_safe / 1000.0)

    # Cross section: sigma = S(E) / E * exp(-B_G / sqrt(E))
    sigma_barns = S_factor / E_safe * np.exp(-B_G / np.sqrt(E_safe))

    # Convert from barns to m^2 (1 barn = 1e-28 m^2)
    sigma = sigma_barns * 1e-28

    # Ensure non-negative
    sigma = np.maximum(sigma, 0.0)

    if sigma.ndim == 0:
        return float(sigma)
    return sigma


def dt_fusion_cross_section(E_keV: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute D-T fusion cross section using Bosch-Hale parameterization.

    Parameters
    ----------
    E_keV : float or ndarray
        Center-of-mass energy [keV].

    Returns
    -------
    float or ndarray
        Cross section [m^2].
    """
    E_keV = np.asarray(E_keV, dtype=float)

    # Bosch-Hale Gamow constant for D-T
    B_G = 34.3827
    
    E_safe = np.maximum(E_keV, 1.0)  # Minimum 1 keV for stability

    # D-T S-factor is larger than D-D due to resonance
    # S(0) ~ 1.1e4 keV-barns with strong energy dependence
    S0 = 1.1e4  # keV-barns
    
    # Simplified S-factor (captures resonance behavior approximately)
    E_res = 64.0  # Resonance energy ~64 keV
    Gamma = 50.0  # Width parameter
    S_factor = S0 / (1.0 + ((E_safe - E_res) / Gamma) ** 2)

    # Cross section: sigma = S(E) / E * exp(-B_G / sqrt(E))
    sigma_barns = S_factor / E_safe * np.exp(-B_G / np.sqrt(E_safe))

    # Convert from barns to m^2
    sigma = sigma_barns * 1e-28

    # Ensure non-negative
    sigma = np.maximum(sigma, 0.0)

    if sigma.ndim == 0:
        return float(sigma)
    return sigma


def bosch_hale_reactivity(
    T_keV: Union[float, np.ndarray],
    reaction: str = "DD",
) -> Union[float, np.ndarray]:
    """Compute thermal reactivity <sigma*v> using Bosch-Hale fit.

    Parameters
    ----------
    T_keV : float or ndarray
        Ion temperature [keV].
    reaction : str
        Reaction type: "DD" or "DT".

    Returns
    -------
    float or ndarray
        Reactivity [m^3/s].
    """
    T_keV = np.asarray(T_keV, dtype=float)
    T_safe = np.maximum(T_keV, 1e-3)

    if reaction.upper() == "DD":
        B_G = 31.3970
        mc2 = 937814.0

        theta = T_safe / (1.0 - T_safe * (0.0 + T_safe * (0.0 + T_safe * 0.0)) / mc2)
        xi = (B_G ** 2 / (4.0 * theta)) ** (1.0 / 3.0)

        C1, C2, C3, C4, C5, C6, C7 = (
            5.43360e-12,
            5.85778e-3,
            7.68222e-3,
            0.0,
            -2.96400e-6,
            0.0,
            0.0,
        )

    else:
        B_G = 34.3827
        mc2 = 1124656.0

        theta = T_safe / (1.0 - T_safe * (0.0 + T_safe * (0.0 + T_safe * 0.0)) / mc2)
        xi = (B_G ** 2 / (4.0 * theta)) ** (1.0 / 3.0)

        C1, C2, C3, C4, C5, C6, C7 = (
            1.17302e-9,
            1.51361e-2,
            7.51886e-2,
            4.60643e-3,
            1.35000e-2,
            -1.06750e-4,
            1.36600e-5,
        )

    numerator = C1 * theta * np.sqrt(xi / (mc2 * T_safe ** 3)) * np.exp(-3.0 * xi)
    denominator = 1.0 + T_safe * (C2 + T_safe * (C4 + T_safe * C6))
    denominator = denominator + T_safe * (C3 + T_safe * (C5 + T_safe * C7))

    reactivity = numerator / denominator

    reactivity = reactivity * 1e-6

    if reactivity.ndim == 0:
        return float(reactivity)
    return reactivity


def beam_target_yield(
    n_beam: float,
    n_target: float,
    E_beam_keV: float,
    volume: float,
    path_length: float,
    reaction: str = "DD",
) -> float:
    """Compute beam-target fusion yield.

    This calculates the fusion yield for a beam of energetic ions
    passing through a thermal target plasma.

    Parameters
    ----------
    n_beam : float
        Beam ion density [m^-3].
    n_target : float
        Target ion density [m^-3].
    E_beam_keV : float
        Beam kinetic energy [keV].
    volume : float
        Interaction volume [m^3].
    path_length : float
        Beam path length [m].
    reaction : str
        Reaction type: "DD" or "DT".

    Returns
    -------
    float
        Total fusion yield [reactions].
    """
    if reaction.upper() == "DD":
        sigma = dd_fusion_cross_section(E_beam_keV)
        m_beam = M_D
    else:
        sigma = dt_fusion_cross_section(E_beam_keV)
        m_beam = M_D

    E_J = E_beam_keV * 1000.0 * q_e
    v_beam = np.sqrt(2.0 * E_J / m_beam)

    rate_density = n_beam * n_target * sigma * v_beam

    yield_total = rate_density * volume

    return float(yield_total)


@dataclass
class BeamTargetModel:
    """Comprehensive beam-target fusion model.

    This class provides a complete model for beam-target fusion
    including beam slowing-down, angular distribution of products,
    and time-integrated yield calculations.

    Parameters
    ----------
    n_target : float
        Target ion density [m^-3].
    T_target_keV : float
        Target ion temperature [keV].
    Z_target : int
        Target ion charge state.
    reaction : str
        Reaction type: "DD" or "DT".
    """

    n_target: float
    T_target_keV: float = 1.0
    Z_target: int = 1
    reaction: str = "DD"

    def cross_section(self, E_keV: float) -> float:
        """Get cross section for the configured reaction."""
        if self.reaction.upper() == "DD":
            return dd_fusion_cross_section(E_keV)
        return dt_fusion_cross_section(E_keV)

    def stopping_power(self, E_keV: float) -> float:
        """Compute beam stopping power in the target.

        Uses a simplified Bethe formula for ion stopping in plasma.

        Parameters
        ----------
        E_keV : float
            Beam energy [keV].

        Returns
        -------
        float
            Stopping power dE/dx [keV/m].
        """
        ln_lambda = 10.0

        v = np.sqrt(2.0 * E_keV * 1000.0 * q_e / M_D)

        dEdx = (
            4.0 * np.pi * q_e ** 4 * self.Z_target ** 2 * self.n_target * ln_lambda
            / (M_D * v ** 2)
        )

        dEdx_keV = dEdx / (1000.0 * q_e)

        return dEdx_keV

    def range(self, E_keV: float, n_steps: int = 100) -> float:
        """Compute beam range in the target.

        Integrates the inverse stopping power to find how far
        the beam travels before thermalizing.

        Parameters
        ----------
        E_keV : float
            Initial beam energy [keV].
        n_steps : int
            Number of integration steps.

        Returns
        -------
        float
            Beam range [m].
        """
        E_min = max(0.01, self.T_target_keV)
        if E_keV <= E_min:
            return 0.0

        energies = np.linspace(E_keV, E_min, n_steps)
        dE = (E_keV - E_min) / n_steps

        total_range = 0.0
        for E in energies:
            dEdx = self.stopping_power(E)
            if dEdx > 0:
                total_range += dE / dEdx

        return total_range

    def yield_during_slowdown(
        self,
        n_beam: float,
        E_initial_keV: float,
        volume: float,
    ) -> float:
        """Compute total yield as beam slows down.

        Integrates the fusion rate over the slowing-down process.

        Parameters
        ----------
        n_beam : float
            Initial beam density [m^-3].
        E_initial_keV : float
            Initial beam energy [keV].
        volume : float
            Interaction volume [m^3].

        Returns
        -------
        float
            Total fusion yield [reactions].
        """
        E_min = max(0.01, self.T_target_keV)
        n_steps = 100

        energies = np.linspace(E_initial_keV, E_min, n_steps)
        dE = (E_initial_keV - E_min) / n_steps

        total_yield = 0.0
        for E in energies:
            sigma = self.cross_section(E)
            dEdx = self.stopping_power(E)
            if dEdx > 0:
                v = np.sqrt(2.0 * E * 1000.0 * q_e / M_D)
                rate = n_beam * self.n_target * sigma * v
                dx = dE / dEdx
                total_yield += rate * volume * dx / v

        return total_yield

    def neutron_energy_spectrum(
        self,
        E_beam_keV: float,
        angles_deg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute neutron energy as function of emission angle.

        For D-D and D-T reactions, the neutron energy depends on
        the emission angle relative to the beam direction.

        Parameters
        ----------
        E_beam_keV : float
            Beam energy [keV].
        angles_deg : ndarray
            Emission angles [degrees].

        Returns
        -------
        tuple of ndarray
            (angles_deg, neutron_energies_keV)
        """
        angles_rad = np.radians(angles_deg)

        E_beam_J = E_beam_keV * 1000.0 * q_e

        if self.reaction.upper() == "DD":
            Q_value = 3.27e6 * q_e
            m_product = 2.809 * AMU
            E_n_cm = 2.45e6 * q_e
        else:
            Q_value = 17.6e6 * q_e
            m_product = 3.727 * AMU
            E_n_cm = 14.1e6 * q_e

        v_cm = np.sqrt(2.0 * E_beam_J / (2.0 * M_D))

        v_n_cm = np.sqrt(2.0 * E_n_cm / (1.675e-27))

        v_n_lab = np.sqrt(v_cm ** 2 + v_n_cm ** 2 + 2 * v_cm * v_n_cm * np.cos(angles_rad))

        E_n_lab = 0.5 * 1.675e-27 * v_n_lab ** 2

        E_n_keV = E_n_lab / (1000.0 * q_e)

        return angles_deg, E_n_keV

    def angular_distribution(
        self,
        E_beam_keV: float,
        angles_deg: np.ndarray,
    ) -> np.ndarray:
        """Compute angular distribution of fusion products.

        Parameters
        ----------
        E_beam_keV : float
            Beam energy [keV].
        angles_deg : ndarray
            Emission angles [degrees].

        Returns
        -------
        ndarray
            Relative yield as function of angle.
        """
        angles_rad = np.radians(angles_deg)

        yield_angular = 0.5 * (1.0 + np.cos(angles_rad) ** 2)

        yield_angular = yield_angular / np.sum(yield_angular)

        return yield_angular
