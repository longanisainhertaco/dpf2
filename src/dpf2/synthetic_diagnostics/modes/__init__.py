"""Synthetic diagnostics for modal analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:  # pragma: no cover - optional dependency for headless testing
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from ...diagnostics.modes import azimuthal_mode_spectrum, growth_rate

__all__ = [
    "azimuthal_mode_spectrum",
    "growth_rate",
    "plot_growth_rates",
    "write_growth_rates",
]


def plot_growth_rates(
    times: Sequence[float],
    spectra: Sequence[Sequence[float]],
    outdir: Path | str = Path("synthetic_diagnostics/modes"),
) -> Path:
    """Plot modal growth rates and return the path to the figure.

    The function expects ``times`` and corresponding mode ``spectra``.  Each
    spectrum must contain the modal amplitudes for a single time.  The data
    are visualised in a simple line plot where the first few modes are
    shown individually.
    """

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = out_path / "growth_rates.png"

    if plt is None:  # pragma: no cover - plotting optional in tests
        # If Matplotlib is unavailable simply write the data to disk so the
        # caller has something to inspect.
        data = np.column_stack([times, np.asarray(spectra)])
        np.savetxt(fig_path.with_suffix(".txt"), data)
        return fig_path

    arr = np.asarray(spectra, dtype=float)
    fig, ax = plt.subplots()
    for m in range(min(arr.shape[1], 4)):
        ax.plot(times, arr[:, m], label=f"m={m}")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def write_growth_rates(
    times: Sequence[float],
    spectra: Sequence[Sequence[float]],
    outdir: Path | str = Path("synthetic_diagnostics/modes"),
) -> Path:
    """Compute growth rates and write them to ``outdir``.

    The returned file contains one row per interval in ``times`` with the
    exponential growth rate for each azimuthal mode.  ``times`` and
    ``spectra`` must therefore contain at least two samples.
    """

    if len(times) < 2 or len(spectra) < 2:
        raise ValueError("at least two time steps required to compute growth rates")

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    rates = []
    for t0, t1, s0, s1 in zip(times[:-1], times[1:], spectra[:-1], spectra[1:]):
        dt = float(t1) - float(t0)
        rates.append(growth_rate(s0, s1, dt))
    arr = np.asarray(rates)
    file_path = out_path / "growth_rates.csv"
    if hasattr(np, "savetxt"):
        np.savetxt(file_path, arr, delimiter=",")
    else:  # pragma: no cover - minimal stub fallback
        with open(file_path, "w", encoding="utf-8") as fh:
            for row in arr:
                fh.write(",".join(str(float(x)) for x in row) + "\n")
    return file_path
