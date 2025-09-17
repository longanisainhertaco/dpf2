"""Plotting utilities for diagnostic outputs.

These helpers load data from HDF5 files produced by the diagnostic classes
and generate simple matplotlib plots.  The functions return the ``Axes`` used
which allows callers to further customise or save the figures.
"""

from __future__ import annotations

from pathlib import Path

try:  # Optional dependency
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("matplotlib is required for plotting diagnostics") from exc

import h5py


def _get_group(h5, name):
    if isinstance(h5, (str, Path)):
        with h5py.File(h5, "r") as f:
            grp = f[name]
            return {k: grp[k][()] for k in grp.keys()}, dict(grp.attrs)
    grp = h5[name]
    return grp, dict(grp.attrs)


def plot_interferometry(h5, name: str, ax=None):
    """Plot phase shift versus time."""
    grp, _ = _get_group(h5, name)
    if ax is None:
        ax = plt.gca()
    ax.plot(grp["time"], grp["phase_shift"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("phase shift")
    return ax


def plot_xray_signal(h5, name: str, ax=None):
    """Plot X-ray detector signal over time."""
    grp, _ = _get_group(h5, name)
    if ax is None:
        ax = plt.gca()
    ax.plot(grp["time"], grp["signal"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("signal")
    return ax


def plot_neutron_tof(h5, name: str, ax=None):
    """Plot neutron time-of-flight histogram."""
    grp, _ = _get_group(h5, name)
    if ax is None:
        ax = plt.gca()
    bins = grp["time_bins"]
    hist = grp["histogram"][0] if grp["histogram"].ndim > 1 else grp["histogram"]
    ax.step(bins[:-1], hist, where="post")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("counts")
    return ax


__all__ = ["plot_interferometry", "plot_xray_signal", "plot_neutron_tof"]
