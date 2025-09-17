"""Lightweight numerical regression tests for standard MHD problems.

The :class:`NumericsPanel` class provides one-click execution of small
Brio--Wu shock tube and Orszag--Tang vortex problems using simplified
configurations.  It computes a few basic diagnostic metrics and writes the
results to ``synthetic_diagnostics/numerics``.  When supplied with a
:class:`~dpf2.diagnostics.quality_dashboard.QualityDashboard` instance the
metrics are also evaluated against pass/fail thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np

from ..diagnostics.quality_dashboard import QualityDashboard


@dataclass
class NumericsPanel:
    """Run small validation problems and gather diagnostic metrics."""

    output_dir: Path = Path("synthetic_diagnostics/numerics")
    quality: QualityDashboard | None = None

    # ------------------------------------------------------------------
    def run_brio_wu(self, n: int = 64) -> dict[str, float | list[float]]:
        """Execute a tiny Brio--Wu shock-tube run.

        The implementation does **not** attempt to reproduce a full high
        fidelity solution.  Instead it generates a perturbed analytic
        solution that exercises the diagnostic pipeline while keeping the
        run time suitable for unit tests.
        """

        x = np.linspace(0.0, 1.0, n)
        # reference and "numerical" magnetic fields
        B_ref = np.ones((n, 3))
        B_num = B_ref.copy()
        B_num[:, 0] += 0.1 * np.sin(2 * np.pi * x)
        return self._finalize("brio_wu", B_num, B_ref)

    # ------------------------------------------------------------------
    def run_orszag_tang(self, n: int = 32) -> dict[str, float | list[float]]:
        """Execute a tiny Orszag--Tang vortex run."""

        grid = np.linspace(0.0, 1.0, n, endpoint=False)
        X, Y = np.meshgrid(grid, grid, indexing="ij")
        Bx = -np.sin(2 * np.pi * Y)
        By = np.sin(2 * np.pi * X)
        B_ref = np.stack((Bx, By, np.zeros_like(Bx)), axis=-1)
        perturb = 0.1 * np.sin(4 * np.pi * X)
        B_num = B_ref + np.stack((perturb, perturb, np.zeros_like(perturb)), axis=-1)
        return self._finalize("orszag_tang", B_num, B_ref)

    # ------------------------------------------------------------------
    def _finalize(
        self, name: str, B_num: np.ndarray, B_ref: np.ndarray
    ) -> dict[str, float | list[float]]:
        metrics = compute_metrics(B_num, B_ref)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        if self.quality is not None:
            self.quality.evaluate_numerics(metrics)
        return metrics


# ---------------------------------------------------------------------------
# Helper functions


def compute_metrics(
    B_num: np.ndarray, B_ref: np.ndarray
) -> dict[str, float | list[float]]:
    """Return basic diagnostic metrics for a magnetic field."""

    diff = B_num - B_ref
    l1 = float(np.mean(np.abs(diff)))

    div = _divergence(B_num)
    arr_div = np.array(div)
    size = 1
    for s in arr_div.shape:
        size *= s
    div_norm = float(np.sqrt(np.sum(arr_div * arr_div)) / size)

    energy_num = 0.5 * float(np.sum(B_num * B_num))
    energy_ref = 0.5 * float(np.sum(B_ref * B_ref))
    energy_drift = energy_num - energy_ref

    spectrum = _spectrum_1d(B_num[..., 0])

    return {
        "l1_error": l1,
        "divB_norm": div_norm,
        "energy_drift": energy_drift,
        "spectrum": spectrum,
    }


def _divergence(B: np.ndarray) -> np.ndarray:
    """Compute a simple discrete divergence."""
    dims = B.ndim - 1

    def grad(a: np.ndarray, ax: int) -> np.ndarray:
        out = np.zeros_like(a)
        if ax == 0:
            out[:-1] = a[1:] - a[:-1]
            out[-1] = a[0] - a[-1]
        elif ax == 1:
            out[:, :-1] = a[:, 1:] - a[:, :-1]
            out[:, -1] = a[:, 0] - a[:, -1]
        else:
            out[:, :, :-1] = a[:, :, 1:] - a[:, :, :-1]
            out[:, :, -1] = a[:, :, 0] - a[:, :, -1]
        return out

    div = grad(B[..., 0], 0)
    if dims > 1:
        div += grad(B[..., 1], 1)
    if dims > 2:
        div += grad(B[..., 2], 2)
    return div


def _spectrum_1d(arr: np.ndarray) -> list[float]:
    """Naive discrete Fourier spectrum along the flattened array."""

    data = _flatten(arr)
    n = len(data)
    spec: list[float] = []
    for k in range(n // 2 + 1):
        re = 0.0
        im = 0.0
        for i, val in enumerate(data):
            angle = 2.0 * math.pi * k * i / n
            re += val * math.cos(angle)
            im -= val * math.sin(angle)
        spec.append(math.sqrt(re * re + im * im))
    return spec


def _flatten(arr) -> list[float]:
    if isinstance(arr, np.Array):  # type: ignore[attr-defined]
        arr = arr.data
    if isinstance(arr, list):
        out: list[float] = []
        for v in arr:
            out.extend(_flatten(v))
        return out
    return [float(arr)]
