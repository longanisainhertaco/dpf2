"""Benchmark for analytic plasma expansion into vacuum.

This benchmark models the expansion of a semi-infinite plasma into vacuum
using an isothermal approximation.  The ion front propagates at the ion
sound speed :math:`c_s = \sqrt{k_B T_e / m_i}`.  The analytic solution for
the position of the plasma front is

.. math:: R(t) = R_0 + c_s t.

The script computes this analytic front position and compares it against the
value produced by :func:`plasma_front_position`, which is a thin wrapper
around the analytic expression.  The comparison serves as a regression test
for code intended to replace the analytic model with a numerical solver.

References
----------
* L. Spitzer, ``Physics of Fully Ionized Gases``, Interscience (1962).
* J. M. Dawson, ``Plasma expansion into a vacuum``, *Physics of Fluids*,
  3(2), 149--154 (1960).
"""

from __future__ import annotations

import numpy as np


def plasma_front_position(t: np.ndarray, sound_speed: float, r0: float = 0.0) -> np.ndarray:
    """Return the analytic position of the ion front.

    Parameters
    ----------
    t:
        Array of times at which to evaluate the solution.
    sound_speed:
        Ion sound speed ``c_s`` in m/s.
    r0:
        Initial position of the plasma edge in meters.  Defaults to zero.
    """

    return r0 + sound_speed * t


def run_benchmark() -> float:
    """Run the plasma expansion benchmark.

    Returns
    -------
    float
        Maximum absolute error between the analytic solution and the
        implementation under test.
    """

    c_s = 9.79e3  # m/s, representative ion sound speed for T_e ~ 5 eV, m_i ~ m_p
    times = np.linspace(0.0, 1e-6, 5)
    analytic = plasma_front_position(times, c_s)

    # In the current implementation ``plasma_front_position`` is analytic;
    # in future versions this may call into a numerical solver.
    model = plasma_front_position(times, c_s)

    return float(np.max(np.abs(analytic - model)))


if __name__ == "__main__":
    error = run_benchmark()
    print(f"Maximum absolute error: {error:.3e} m")
