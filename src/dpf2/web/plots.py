from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_current_voltage(
    t: Sequence[float],
    current: Sequence[float],
    voltage: Sequence[float],
    path: str | Path,
) -> Path:
    """Overlay current and voltage versus time on shared axes."""
    fig, ax1 = plt.subplots()
    ax1.plot(t, current, color="tab:blue", label="current")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (A)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(t, voltage, color="tab:red", label="voltage")
    ax2.set_ylabel("Voltage (V)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_vector_field_overlay(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    path: str | Path,
    background: str | Path | None = None,
) -> Path:
    """Create a simple vector field overlay with an optional background image."""
    fig, ax = plt.subplots()
    if background and Path(background).exists():
        img = plt.imread(background)
        ax.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
    ax.quiver(x, y, u, v, color="tab:green")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = ["plot_current_voltage", "plot_vector_field_overlay"]
