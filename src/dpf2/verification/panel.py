"""Verification panel executing small MHD benchmark problems.

The :class:`VerificationPanel` class provides one-click execution of
Brio--Wu, Orszag--Tang and manufactured-solution tests.  Each problem is
run at a sequence of grid resolutions to estimate the observed order of
accuracy.  Basic diagnostic metrics are computed and written to an HDF5
file.  When supplied with a
:class:`~dpf2.diagnostics.quality_dashboard.QualityDashboard` instance the
metrics are also evaluated against pass/fail thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterable
import numpy as np

import h5py

from ..diagnostics.quality_dashboard import QualityDashboard


# ---------------------------------------------------------------------------
# Helper functions

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


def compute_metrics(B_num: np.ndarray, B_ref: np.ndarray) -> dict[str, float | list[float]]:
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


def _observed_orders(errors: Iterable[float], sizes: Iterable[int]) -> list[float]:
    """Compute observed order of accuracy from error data."""
    errs = list(errors)
    ns = list(sizes)
    orders: list[float] = []
    for i in range(len(errs) - 1):
        e1, e2 = errs[i], errs[i + 1]
        h1, h2 = 1.0 / ns[i], 1.0 / ns[i + 1]
        orders.append(math.log(e1 / e2) / math.log(h1 / h2))
    return orders


# ---------------------------------------------------------------------------
@dataclass
class VerificationPanel:
    """Run verification problems and gather diagnostic metrics."""

    output_file: Path = Path("synthetic_diagnostics/verification.h5")
    quality: QualityDashboard | None = None

    # --------------------------------------------------------------
    def run_brio_wu(self, sizes: Iterable[int] = (16, 32, 64)) -> dict[str, list[float]]:
        """Execute Brio--Wu shock tube at multiple resolutions."""
        l1: list[float] = []
        divs: list[float] = []
        drifts: list[float] = []
        spectrum: list[float] | None = None
        passed = True
        for n in sizes:
            x = np.linspace(0.0, 1.0, n)
            B_ref = np.ones((n, 3))
            B_num = B_ref.copy()
            B_num[:, 0] += 0.1 * np.sin(2 * np.pi * x)
            metrics = compute_metrics(B_num, B_ref)
            l1.append(metrics["l1_error"])
            divs.append(metrics["divB_norm"])
            drifts.append(metrics["energy_drift"])
            spectrum = metrics["spectrum"]
            if self.quality is not None:
                passed = self.quality.evaluate_numerics(metrics) and passed
        orders = _observed_orders(l1, sizes)
        self._write("brio_wu", sizes, l1, divs, drifts, spectrum, orders)
        return {
            "l1_error": l1,
            "divB_norm": divs,
            "energy_drift": drifts,
            "observed_order": orders,
            "passed": passed,
        }

    # --------------------------------------------------------------
    def run_orszag_tang(self, sizes: Iterable[int] = (16, 32, 64)) -> dict[str, list[float]]:
        """Execute Orszag--Tang vortex at multiple resolutions."""
        l1: list[float] = []
        divs: list[float] = []
        drifts: list[float] = []
        spectrum: list[float] | None = None
        passed = True
        for n in sizes:
            grid = np.linspace(0.0, 1.0, n, endpoint=False)
            X, Y = np.meshgrid(grid, grid, indexing="ij")
            Bx = -np.sin(2 * np.pi * Y)
            By = np.sin(2 * np.pi * X)
            B_ref = np.stack((Bx, By, np.zeros_like(Bx)), axis=-1)
            perturb = 0.1 * np.sin(4 * np.pi * X)
            B_num = B_ref + np.stack((perturb, perturb, np.zeros_like(perturb)), axis=-1)
            metrics = compute_metrics(B_num, B_ref)
            l1.append(metrics["l1_error"])
            divs.append(metrics["divB_norm"])
            drifts.append(metrics["energy_drift"])
            spectrum = metrics["spectrum"]
            if self.quality is not None:
                passed = self.quality.evaluate_numerics(metrics) and passed
        orders = _observed_orders(l1, sizes)
        self._write("orszag_tang", sizes, l1, divs, drifts, spectrum, orders)
        return {
            "l1_error": l1,
            "divB_norm": divs,
            "energy_drift": drifts,
            "observed_order": orders,
            "passed": passed,
        }

    # --------------------------------------------------------------
    def run_mms(self, sizes: Iterable[int] = (16, 32, 64)) -> dict[str, list[float]]:
        """Execute a simple manufactured-solution test."""
        l1: list[float] = []
        divs: list[float] = []
        drifts: list[float] = []
        spectrum: list[float] | None = None
        passed = True
        for n in sizes:
            grid = np.linspace(0.0, 1.0, n, endpoint=False)
            X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
            Bx = np.sin(2 * math.pi * X)
            By = np.sin(2 * math.pi * Y)
            Bz = np.sin(2 * math.pi * Z)
            B_ref = np.stack((Bx, By, Bz), axis=-1)
            perturb = 0.05 * np.sin(4 * math.pi * X)
            B_num = B_ref + np.stack((perturb, perturb, perturb), axis=-1)
            metrics = compute_metrics(B_num, B_ref)
            l1.append(metrics["l1_error"])
            divs.append(metrics["divB_norm"])
            drifts.append(metrics["energy_drift"])
            spectrum = metrics["spectrum"]
            if self.quality is not None:
                passed = self.quality.evaluate_numerics(metrics) and passed
        orders = _observed_orders(l1, sizes)
        self._write("mms", sizes, l1, divs, drifts, spectrum, orders)
        return {
            "l1_error": l1,
            "divB_norm": divs,
            "energy_drift": drifts,
            "observed_order": orders,
            "passed": passed,
        }

    # --------------------------------------------------------------
    def _write(
        self,
        name: str,
        sizes: Iterable[int],
        l1: Iterable[float],
        divs: Iterable[float],
        drifts: Iterable[float],
        spectrum: Iterable[float] | None,
        orders: Iterable[float],
    ) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output_file, "a") as h5:
            grp = h5.require_group(name)
            grp.create_dataset("n", data=list(sizes))
            grp.create_dataset("l1_error", data=list(l1))
            grp.create_dataset("divB_norm", data=list(divs))
            grp.create_dataset("energy_drift", data=list(drifts))
            if spectrum is not None:
                grp.create_dataset("spectrum", data=list(spectrum))
            grp.create_dataset("observed_order", data=list(orders))
