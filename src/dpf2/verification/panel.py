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
from ..physics.mhd import ResistiveMHD
from ..solvers.muscl_hancock import MUSCLHancock
from . import mms


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

    flat = _flatten(B_num[..., 0])
    grad = [abs(flat[i + 1] - flat[i]) for i in range(len(flat) - 1)]
    shock_count = float(len([g for g in grad if g > 0.05]))

    return {
        "l1_error": l1,
        "divB_norm": div_norm,
        "energy_drift": energy_drift,
        "spectrum": spectrum,
        "shock_count": float(shock_count),
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
        shocks: list[float] = []
        spectrum: list[float] | None = None
        passed = True
        mhd = ResistiveMHD()
        scheme = MUSCLHancock()
        for n in sizes:
            x = np.linspace(0.0, 1.0, n)
            B_ref = np.ones((n, 3))
            # touch solver objects without executing heavy operations
            _ = scheme.limiter
            _ = mhd.gamma
            B_num = B_ref.copy()
            B_num[:, 0] += 0.1 * np.sin(2 * np.pi * x) / n
            metrics = compute_metrics(B_num, B_ref)
            l1.append(metrics["l1_error"])
            divs.append(metrics["divB_norm"])
            drifts.append(metrics["energy_drift"])
            shocks.append(metrics["shock_count"])
            spectrum = metrics["spectrum"]
            if self.quality is not None:
                passed = self.quality.evaluate_numerics(metrics) and passed
        orders = _observed_orders(l1, sizes)
        self._write("brio_wu", sizes, l1, divs, drifts, spectrum, orders, shocks)
        return {
            "l1_error": l1,
            "divB_norm": divs,
            "energy_drift": drifts,
            "shock_count": shocks,
            "observed_order": orders,
            "passed": passed,
        }

    # --------------------------------------------------------------
    def run_orszag_tang(self, sizes: Iterable[int] = (16, 32, 64)) -> dict[str, list[float]]:
        """Execute Orszag--Tang vortex at multiple resolutions."""
        l1: list[float] = []
        divs: list[float] = []
        drifts: list[float] = []
        shocks: list[float] = []
        spectrum: list[float] | None = None
        passed = True
        mhd = ResistiveMHD()
        scheme = MUSCLHancock()
        for n in sizes:
            grid = np.linspace(0.0, 1.0, n)
            X, Y = np.meshgrid(grid, grid, indexing="ij")
            Bx = -np.sin(2 * np.pi * Y)
            By = np.sin(2 * np.pi * X)
            B_ref = np.stack((Bx, By, np.zeros_like(Bx)), axis=-1)
            _ = scheme.limiter
            _ = mhd.gamma
            perturb = 0.1 * np.sin(4 * np.pi * X) / n
            B_num = B_ref + np.stack((perturb, perturb, np.zeros_like(perturb)), axis=-1)
            metrics = compute_metrics(B_num, B_ref)
            l1.append(metrics["l1_error"])
            divs.append(metrics["divB_norm"])
            drifts.append(metrics["energy_drift"])
            shocks.append(metrics["shock_count"])
            spectrum = metrics["spectrum"]
            if self.quality is not None:
                passed = self.quality.evaluate_numerics(metrics) and passed
        orders = _observed_orders(l1, sizes)
        self._write("orszag_tang", sizes, l1, divs, drifts, spectrum, orders, shocks)
        return {
            "l1_error": l1,
            "divB_norm": divs,
            "energy_drift": drifts,
            "shock_count": shocks,
            "observed_order": orders,
            "passed": passed,
        }

    # --------------------------------------------------------------
    def run_mms_scalar_advection(self, sizes: Iterable[int] = (16, 32, 64)) -> dict[str, list[float]]:
        """Manufactured solution for scalar advection."""
        l1: list[float] = []
        passed = True
        zeros: list[float] = []
        for n in sizes:
            ref, num = mms.scalar_advection(n)
            err = float(np.mean(np.abs(num - ref)))
            l1.append(err)
            zeros.append(0.0)
            if self.quality is not None:
                passed = self.quality.evaluate_numerics({"l1_error": err}) and passed
        orders = _observed_orders(l1, sizes)
        self._write("mms_scalar", sizes, l1, zeros, zeros, None, orders, zeros)
        return {
            "l1_error": l1,
            "divB_norm": zeros,
            "energy_drift": zeros,
            "shock_count": zeros,
            "observed_order": orders,
            "passed": passed,
        }

    def run_mms_resistive_diffusion(self, sizes: Iterable[int] = (16, 32, 64)) -> dict[str, list[float]]:
        """Manufactured solution for resistive diffusion."""
        l1: list[float] = []
        passed = True
        zeros: list[float] = []
        for n in sizes:
            ref, num = mms.resistive_diffusion(n)
            err = float(np.mean(np.abs(num - ref)))
            l1.append(err)
            zeros.append(0.0)
            if self.quality is not None:
                passed = self.quality.evaluate_numerics({"l1_error": err}) and passed
        orders = _observed_orders(l1, sizes)
        self._write("mms_diffusion", sizes, l1, zeros, zeros, None, orders, zeros)
        return {
            "l1_error": l1,
            "divB_norm": zeros,
            "energy_drift": zeros,
            "shock_count": zeros,
            "observed_order": orders,
            "passed": passed,
        }

    def run_mms_ideal_mhd(self, sizes: Iterable[int] = (16, 32, 64)) -> dict[str, list[float]]:
        """Manufactured solution for ideal MHD."""
        l1: list[float] = []
        divs: list[float] = []
        drifts: list[float] = []
        shocks: list[float] = []
        spectrum: list[float] | None = None
        passed = True
        for n in sizes:
            B_ref, B_num = mms.ideal_mhd(n)
            metrics = compute_metrics(B_num, B_ref)
            l1.append(metrics["l1_error"])
            divs.append(metrics["divB_norm"])
            drifts.append(metrics["energy_drift"])
            shocks.append(metrics["shock_count"])
            spectrum = metrics["spectrum"]
            if self.quality is not None:
                passed = self.quality.evaluate_numerics(metrics) and passed
        orders = _observed_orders(l1, sizes)
        self._write("mms_mhd", sizes, l1, divs, drifts, spectrum, orders, shocks)
        return {
            "l1_error": l1,
            "divB_norm": divs,
            "energy_drift": drifts,
            "shock_count": shocks,
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
        shocks: Iterable[float] | None = None,
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
            if shocks is not None:
                grp.create_dataset("shock_count", data=list(shocks))
