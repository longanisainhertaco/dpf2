import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

from circuit_config import CircuitConfig


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
