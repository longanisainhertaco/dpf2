"""Whistler wave dispersion relation for Hall-MHD plasmas.

This module provides functions for computing whistler wave properties
in magnetized plasmas, which are important for understanding Hall-MHD
dynamics at small scales.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

try:
    from scipy.constants import mu_0, e as q_e, m_e, m_p, c as c_light
except ImportError:
    mu_0 = 4e-7 * np.pi
    q_e = 1.602176634e-19
    m_e = 9.1093837015e-31
    m_p = 1.67262192369e-27
    c_light = 299792458.0

__all__ = [
    "whistler_frequency",
    "dispersion_relation",
    "phase_velocity",
    "group_velocity",
    "critical_wavelength",
]


def ion_inertial_length(n_e: float) -> float:
    """Compute ion inertial length d_i = c/omega_pi.

    Parameters
    ----------
    n_e : float
        Electron density [m^-3].

    Returns
    -------
    float
        Ion inertial length [m].
    """
    # omega_pi = sqrt(n_i * q_e^2 / (m_p * epsilon_0)) where epsilon_0 = 1/(mu_0 * c^2)
    # Simplifies to: omega_pi = sqrt(n_e * q_e^2 * mu_0 * c^2 / m_p) = c * sqrt(n_e * q_e^2 * mu_0 / m_p)
    epsilon_0 = 1.0 / (mu_0 * c_light ** 2)
    omega_pi = np.sqrt(n_e * q_e ** 2 / (m_p * epsilon_0))
    return c_light / max(omega_pi, 1e-30)


def ion_cyclotron_frequency(B: float) -> float:
    """Compute ion cyclotron frequency omega_ci = eB/m_i.

    Parameters
    ----------
    B : float
        Magnetic field strength [T].

    Returns
    -------
    float
        Ion cyclotron frequency [rad/s].
    """
    return abs(q_e) * B / m_p


def electron_cyclotron_frequency(B: float) -> float:
    """Compute electron cyclotron frequency omega_ce = eB/m_e.

    Parameters
    ----------
    B : float
        Magnetic field strength [T].

    Returns
    -------
    float
        Electron cyclotron frequency [rad/s].
    """
    return abs(q_e) * B / m_e


def whistler_frequency(
    k: Union[float, np.ndarray],
    n_e: float,
    B: float,
    theta: float = 0.0,
) -> Union[float, np.ndarray]:
    """Compute whistler wave frequency from dispersion relation.

    The whistler wave frequency in the limit k*d_i >> 1 is given by:

        omega = omega_ci * (k * d_i)^2 * cos(theta)

    where d_i is the ion inertial length and theta is the angle
    between k and B.

    Parameters
    ----------
    k : float or ndarray
        Wavenumber [rad/m].
    n_e : float
        Electron density [m^-3].
    B : float
        Magnetic field strength [T].
    theta : float
        Angle between k and B [radians], default 0 (parallel).

    Returns
    -------
    float or ndarray
        Whistler wave frequency [rad/s].
    """
    d_i = ion_inertial_length(n_e)
    omega_ci = ion_cyclotron_frequency(B)

    k = np.asarray(k)
    omega = omega_ci * (k * d_i) ** 2 * np.cos(theta)

    if omega.ndim == 0:
        return float(omega)
    return omega


def dispersion_relation(
    k: Union[float, np.ndarray],
    n_e: float,
    B: float,
    theta: float = 0.0,
    include_electron_inertia: bool = False,
) -> Union[float, np.ndarray]:
    """Full whistler dispersion relation including corrections.

    The full dispersion relation for whistler waves in Hall-MHD is:

        omega = k^2 * d_i^2 * omega_ci * cos(theta) / (1 + k^2 * d_e^2)

    where d_e is the electron inertial length.

    Parameters
    ----------
    k : float or ndarray
        Wavenumber [rad/m].
    n_e : float
        Electron density [m^-3].
    B : float
        Magnetic field strength [T].
    theta : float
        Angle between k and B [radians].
    include_electron_inertia : bool
        If True, include electron inertia correction.

    Returns
    -------
    float or ndarray
        Wave frequency [rad/s].
    """
    d_i = ion_inertial_length(n_e)
    omega_ci = ion_cyclotron_frequency(B)

    k = np.asarray(k)
    k_di = k * d_i

    omega = omega_ci * k_di ** 2 * np.cos(theta)

    if include_electron_inertia:
        omega_pe = np.sqrt(n_e * q_e ** 2 / (m_e * mu_0 * c_light ** 2)) * c_light
        d_e = c_light / max(omega_pe, 1e-30)
        k_de = k * d_e
        omega = omega / (1.0 + k_de ** 2)

    if omega.ndim == 0:
        return float(omega)
    return omega


def phase_velocity(
    k: Union[float, np.ndarray],
    n_e: float,
    B: float,
    theta: float = 0.0,
) -> Union[float, np.ndarray]:
    """Compute whistler wave phase velocity v_ph = omega/k.

    Parameters
    ----------
    k : float or ndarray
        Wavenumber [rad/m].
    n_e : float
        Electron density [m^-3].
    B : float
        Magnetic field strength [T].
    theta : float
        Angle between k and B [radians].

    Returns
    -------
    float or ndarray
        Phase velocity [m/s].
    """
    omega = whistler_frequency(k, n_e, B, theta)
    k = np.asarray(k)
    v_ph = omega / np.maximum(k, 1e-30)

    if v_ph.ndim == 0:
        return float(v_ph)
    return v_ph


def group_velocity(
    k: Union[float, np.ndarray],
    n_e: float,
    B: float,
    theta: float = 0.0,
) -> Union[float, np.ndarray]:
    """Compute whistler wave group velocity v_g = d(omega)/dk.

    For the simple whistler dispersion omega = omega_ci * (k*d_i)^2,
    the group velocity is:

        v_g = 2 * omega_ci * d_i^2 * k * cos(theta)

    Parameters
    ----------
    k : float or ndarray
        Wavenumber [rad/m].
    n_e : float
        Electron density [m^-3].
    B : float
        Magnetic field strength [T].
    theta : float
        Angle between k and B [radians].

    Returns
    -------
    float or ndarray
        Group velocity [m/s].
    """
    d_i = ion_inertial_length(n_e)
    omega_ci = ion_cyclotron_frequency(B)

    k = np.asarray(k)
    v_g = 2.0 * omega_ci * d_i ** 2 * k * np.cos(theta)

    if v_g.ndim == 0:
        return float(v_g)
    return v_g


def critical_wavelength(n_e: float, B: float) -> float:
    """Compute critical wavelength where Hall effects become important.

    Hall physics becomes important when the wavelength is comparable
    to or smaller than the ion inertial length d_i.

    Parameters
    ----------
    n_e : float
        Electron density [m^-3].
    B : float
        Magnetic field strength [T].

    Returns
    -------
    float
        Critical wavelength [m].
    """
    return 2.0 * np.pi * ion_inertial_length(n_e)


def max_whistler_frequency(n_e: float, B: float) -> float:
    """Compute maximum whistler frequency (at electron cyclotron).

    Whistler waves exist below the electron cyclotron frequency.

    Parameters
    ----------
    n_e : float
        Electron density [m^-3].
    B : float
        Magnetic field strength [T].

    Returns
    -------
    float
        Maximum whistler frequency [rad/s].
    """
    return electron_cyclotron_frequency(B)


def whistler_damping_rate(
    k: float,
    n_e: float,
    B: float,
    Te: float,
    collision_frequency: Optional[float] = None,
) -> float:
    """Estimate whistler wave damping rate.

    Parameters
    ----------
    k : float
        Wavenumber [rad/m].
    n_e : float
        Electron density [m^-3].
    B : float
        Magnetic field strength [T].
    Te : float
        Electron temperature [K].
    collision_frequency : float, optional
        Electron collision frequency [1/s].

    Returns
    -------
    float
        Damping rate [1/s].
    """
    omega_ce = electron_cyclotron_frequency(B)

    if collision_frequency is not None:
        nu = collision_frequency
    else:
        try:
            from scipy.constants import k as k_B
        except ImportError:
            k_B = 1.380649e-23
        ln_lambda = 10.0
        nu = 2.91e-6 * n_e * ln_lambda / (Te ** 1.5)

    gamma = 0.5 * nu * (1.0 - whistler_frequency(k, n_e, B) / omega_ce)
    return max(gamma, 0.0)
