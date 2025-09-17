from __future__ import annotations

"""Utilities for analyzing axial rundown behavior.

This module provides helpers to compute the dimensionless shock
parameter ``S`` defined by::

    S = I / (a * p0)

where ``I`` is the discharge current, ``a`` is the anode radius and
``p0`` is the fill gas pressure.  ``S`` is a useful metric for
characterizing rundown similarity across devices.  A simple plotting
utility is also provided for quick‑look diagnostics.
"""

from pathlib import Path
from typing import Iterable

import numpy as np

try:  # pragma: no cover - plotting is optional
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - used when matplotlib is absent
    plt = None  # type: ignore


def shock_parameter(
    current: Iterable[float], anode_radius: float, p0: float
) -> np.ndarray:
    """Return the dimensionless shock parameter ``S``.

    Parameters
    ----------
    current:
        Iterable of discharge current values in amperes.
    anode_radius:
        Anode radius ``a`` in metres.
    p0:
        Fill gas pressure ``p0`` in pascals.
    """

    I = np.array(list(current))
    if anode_radius <= 0 or p0 <= 0:
        raise ValueError("anode_radius and p0 must be positive")
    return I / (anode_radius * p0)


def plot_shock_parameter(time: Iterable[float], S: Iterable[float], path: Path) -> Path:
    """Plot ``S`` versus ``time`` and write to ``path``.

    Parameters
    ----------
    time:
        Iterable of time samples in seconds.
    S:
        Iterable of ``S`` values as returned by :func:`shock_parameter`.
    path:
        Output image file.  Directories are created as needed.
    """

    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")

    t = np.array(list(time))
    s = np.array(list(S))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(t, s)
    plt.xlabel("Time (s)")
    plt.ylabel("S = I/(a*p0)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


__all__ = ["shock_parameter", "plot_shock_parameter"]
