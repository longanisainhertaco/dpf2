"""Simple RLC solver with optional dynamic plasma contributions.

The helper functions implemented here provide crude estimates of the
additional inductance and resistance introduced by the plasma column in a
Dense Plasma Focus device.  They are intentionally lightweight and
analytic so that unit tests can exercise the coupling between the pinch
model and the circuit without requiring a full MHD simulation.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

from circuit_config import CircuitConfig


MU_0 = 4e-7 * np.pi


def dynamic_inductance(radius: np.ndarray, cathode_radius: float, length: float = 0.1) -> np.ndarray:
    """Estimate plasma inductance from pinch radius.

    Parameters
    ----------
    radius:
        Instantaneous pinch radius [m].  Can be a scalar or array.
    cathode_radius:
        Cathode radius of the device [m].
    length:
        Effective length of the plasma column [m].

    Returns
    -------
    ndarray
        Plasma inductance [H] for each radius value.

    Notes
    -----
    A simple coaxial geometry is assumed with the inductance given by
    ``L = mu0 * length / (2*pi) * ln(b/r)``.  Radii are clipped to a
    small positive value to avoid singularities as the pinch collapses.
    """

    r = np.clip(np.asarray(radius), 1e-9, None)
    if cathode_radius <= 0 or length <= 0:
        raise ValueError("cathode_radius and length must be positive")
    return MU_0 * length / (2 * np.pi) * np.log(cathode_radius / r)


def dynamic_resistance(
    radius: np.ndarray,
    density: np.ndarray,
    temperature: np.ndarray,
    length: float = 0.1,
    ln_lambda: float = 10.0,
    z_eff: float = 1.0,
) -> np.ndarray:
    """Crude Spitzer-like plasma resistance estimate.

    Parameters
    ----------
    radius:
        Pinch radius [m].
    density:
        Plasma mass density [kg/m^3].
    temperature:
        Electron temperature [K].
    length:
        Plasma column length [m].
    ln_lambda:
        Coulomb logarithm used in the resistivity estimate.
    z_eff:
        Effective charge state.

    Returns
    -------
    ndarray
        Plasma resistance [Ω] corresponding to the supplied state.

    Notes
    -----
    The Spitzer resistivity is approximated by ``eta ≈ 1.65e-9 * Z *
    lnΛ / T_e[eV]^{3/2}`` (Ω·m).  The resistance is then ``eta * length /
    area`` where ``area = π r^2``.  Temperatures supplied in Kelvin are
    converted to electron-volts using ``1 eV ≈ 11604 K``.
    """

    r = np.clip(np.asarray(radius), 1e-9, None)
    rho = np.clip(np.asarray(density), 1e-30, None)
    T = np.clip(np.asarray(temperature), 1e-9, None)

    area = np.pi * r ** 2
    # Convert temperature to eV for Spitzer formula
    T_eV = T / 11604.0
    n_e = rho / 1.6726219e-27  # electron density assuming deuterium
    eta = 1.65e-9 * z_eff * ln_lambda / (np.power(T_eV, 1.5) * np.maximum(n_e, 1e6))
    return eta * length / area


def run_circuit_simulation(cfg: CircuitConfig, t_end: float, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run simple RLC discharge using cfg parameters.

    Parameters
    ----------
    cfg : CircuitConfig
        Circuit configuration with L_ext [uH], R_ext [mOhm], C_ext [uF], V0 [kV].
    t_end : float
        End time in microseconds.
    num_points : int, optional
        Number of output points, by default 1000.

    Returns
    -------
    tuple of ndarray
        time [s], current [A], capacitor voltage [V].
    """
    # convert to SI units
    L = cfg.L_ext * 1e-6
    R = cfg.R_ext * 1e-3
    C = cfg.C_ext * 1e-6
    V0 = cfg.V0 * 1e3
    delay = cfg.switch_delay * 1e-9

    def rlc_ode(t, y):
        I, Q = y
        dIdt = -(R / L) * I - Q / (L * C)
        dQdt = I
        return [dIdt, dQdt]

    # time grid in seconds
    t_total = np.linspace(0.0, t_end * 1e-6, num_points)

    if t_end * 1e-6 <= delay:
        # switch never closes
        current = np.zeros_like(t_total)
        voltage = np.full_like(t_total, V0)
        return t_total, current, voltage

    # before switch closes
    mask_before = t_total < delay
    current = np.zeros_like(t_total)
    voltage = np.full_like(t_total, V0)

    # integrate after delay
    t_eval = t_total[~mask_before]
    q0 = C * V0
    sol = solve_ivp(rlc_ode, (delay, t_total[-1]), [0.0, q0], t_eval=t_eval, method="RK45")
    current[~mask_before] = sol.y[0]
    voltage[~mask_before] = sol.y[1] / C

    return t_total, current, voltage
