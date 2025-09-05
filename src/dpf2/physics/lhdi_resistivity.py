from __future__ import annotations

"""Lower-hybrid drift wave derived effective resistivity utilities.

This module exposes :func:`compute_effective_eta` which estimates an
anomalous resistivity and corresponding axial electric-field surge from
simple lower-hybrid drift wave diagnostics.  The implementation is
minimal and primarily intended to provide a lightweight stand-in for
more sophisticated models.
"""

from typing import Any, Tuple
import numpy as np


def _to_array(val: Any) -> np.ndarray:
    """Return ``val`` as a floating point array.

    Tolerates the lightweight ``numpy`` substitute used in the tests, which
    lacks support for the ``dtype`` argument.
    """

    try:  # pragma: no cover - real ``numpy`` path
        return np.asarray(val, dtype=float)
    except TypeError:  # pragma: no cover - ``numpy_stub`` path
        return np.asarray(val)


def compute_effective_eta(wave_power: Any, phase_velocity: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return an effective resistivity and axial electric-field surge.

    Parameters
    ----------
    wave_power:
        Turbulent wave power driving anomalous transport [W m^-3].
    phase_velocity:
        Estimated wave phase velocity [m s^-1].

    Returns
    -------
    eta:
        Effective anomalous resistivity [Ohm m].
    e_field:
        Axial electric-field surge associated with the wave [V m^-1].

    Notes
    -----
    The resistivity is approximated by ``wave_power / phase_velocity**2`` and
    the axial electric field is ``eta * phase_velocity``.  Inputs are converted
    to :class:`numpy.ndarray` and broadcast following NumPy semantics.  A small
    numerical floor prevents division by zero when ``phase_velocity`` contains
    zeros.
    """

    power = _to_array(wave_power)
    velocity = _to_array(phase_velocity)

    eta = power / (velocity ** 2 + 1.0e-30)
    e_field = eta * velocity
    return eta, e_field


__all__ = ["compute_effective_eta"]
