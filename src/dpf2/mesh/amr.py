from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import pyamrex  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade if backend missing
    pyamrex = None  # type: ignore


# ---------------------------------------------------------------------------
# Refinement criteria helpers

def plasma_gradient_refinement(field: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask of cells where the gradient magnitude exceeds ``threshold``."""
    if field is None:
        raise ValueError("field must be provided")
    arr = np.asarray(field)
    shape = arr.shape
    if arr.ndim == 1:
        mask = [False] * shape[0]
        for i in range(shape[0]):
            left = arr[i - 1 if i > 0 else 0]
            right = arr[i + 1 if i < shape[0] - 1 else shape[0] - 1]
            gx = float(right) - float(left)
            mag = abs(gx) * 0.5
            mask[i] = mag > threshold
    else:
        mask = [[False] * shape[1] for _ in range(shape[0])]
        for i in range(shape[0]):
            for j in range(shape[1]):
                left = arr[i - 1 if i > 0 else 0, j]
                right = arr[i + 1 if i < shape[0] - 1 else shape[0] - 1, j]
                down = arr[i, j - 1 if j > 0 else 0]
                up = arr[i, j + 1 if j < shape[1] - 1 else shape[1] - 1]
                gx = float(right) - float(left)
                gy = float(up) - float(down)
                mag = (gx ** 2 + gy ** 2) ** 0.5 * 0.5
                mask[i][j] = mag > threshold
    return np.array(mask)


def debye_length_refinement(lambda_D: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where the Debye length ``lambda_D`` falls below ``threshold``."""
    arr = lambda_D.data if hasattr(lambda_D, "data") else lambda_D
    def _walk(a):
        if isinstance(a, list):
            return [_walk(x) for x in a]
        return bool(a < threshold)
    return np.array(_walk(arr))


def ion_inertial_length_refinement(d_i: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where the ion inertial length ``d_i`` falls below ``threshold``."""
    arr = d_i.data if hasattr(d_i, "data") else d_i
    def _walk(a):
        if isinstance(a, list):
            return [_walk(x) for x in a]
        return bool(a < threshold)
    return np.array(_walk(arr))


def pressure_gradient_refinement(pressure: np.ndarray, threshold: float) -> np.ndarray:
    """Convenience wrapper applying :func:`plasma_gradient_refinement` to pressure."""
    return plasma_gradient_refinement(pressure, threshold)


def current_density_refinement(current: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where the current density magnitude exceeds ``threshold``."""
    arr = current.data if hasattr(current, "data") else current
    def _walk(a):
        if isinstance(a, list) and a and not isinstance(a[0], list):
            if len(a) != 3:
                raise ValueError("current vectors must have three components")
            mag = (a[0]**2 + a[1]**2 + a[2]**2) ** 0.5
            return mag > threshold
        if isinstance(a, list):
            return [_walk(x) for x in a]
        raise ValueError("current must be an array of vectors")
    return np.array(_walk(arr))


def wavefront_refinement(field: np.ndarray, prev_field: np.ndarray, threshold: float) -> np.ndarray:
    """Tag cells where the change between two fields exceeds ``threshold``."""
    if field is None or prev_field is None:
        raise ValueError("Both current and previous fields are required")
    arr = np.asarray(field)
    prev = np.asarray(prev_field)
    shape = arr.shape
    if arr.ndim == 1:
        mask = [False] * shape[0]
        for i in range(shape[0]):
            if abs(float(arr[i]) - float(prev[i])) > threshold:
                mask[i] = True
    else:
        mask = [[False] * shape[1] for _ in range(shape[0])]
        for i in range(shape[0]):
            for j in range(shape[1]):
                if abs(float(arr[i, j]) - float(prev[i, j])) > threshold:
                    mask[i][j] = True
    return np.array(mask)


@dataclass
class AMRMesh:
    """Light‑weight wrapper around a pyAMReX style AMR interface."""

    shape: tuple[int, ...]
    criteria: Dict[str, float]

    def __post_init__(self) -> None:
        self._backend = pyamrex
        self._last_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def refine(
        self,
        plasma_state: Optional[Dict[str, Any]] = None,
        prev_field: Optional[np.ndarray] = None,
    ) -> None:
        """Apply refinement criteria and notify the backend."""
        if plasma_state is None:
            return

        density = plasma_state.get("density")
        field = plasma_state.get("field")
        lambda_D = plasma_state.get("lambda_D")
        d_i = plasma_state.get("d_i")
        pressure = plasma_state.get("pressure")
        current = plasma_state.get("current")

        def _init_mask(shape):
            if isinstance(shape, int):
                return [False] * shape
            if isinstance(shape, tuple):
                return [_init_mask(s) for s in shape]
            raise TypeError("shape must be int or tuple")

        def _or(a, b):
            if isinstance(a, list):
                return [_or(x, y) for x, y in zip(a, b)]
            return bool(a) or bool(b)

        mask = _init_mask(self.shape)

        grad_thresh = self.criteria.get("gradient_threshold")
        if density is not None and grad_thresh is not None:
            mask = _or(mask, plasma_gradient_refinement(density, grad_thresh).data)

        wave_thresh = self.criteria.get("wavefront_threshold")
        if field is not None and prev_field is not None and wave_thresh is not None:
            mask = _or(mask, wavefront_refinement(field, prev_field, wave_thresh).data)

        ld_thresh = self.criteria.get("lambda_D_threshold")
        if lambda_D is not None and ld_thresh is not None:
            mask = _or(mask, debye_length_refinement(lambda_D, ld_thresh).data)

        di_thresh = self.criteria.get("d_i_threshold")
        if d_i is not None and di_thresh is not None:
            mask = _or(mask, ion_inertial_length_refinement(d_i, di_thresh).data)

        gradp_thresh = self.criteria.get("pressure_gradient_threshold")
        if pressure is not None and gradp_thresh is not None:
            mask = _or(mask, pressure_gradient_refinement(pressure, gradp_thresh).data)

        J_thresh = self.criteria.get("current_density_threshold")
        if current is not None and J_thresh is not None:
            mask = _or(mask, current_density_refinement(current, J_thresh).data)

        self._last_mask = np.array(mask)

        if self._backend and hasattr(self._backend, "amr"):
            self._backend.amr.tag_cells(mask.astype(np.int8))
            if hasattr(self._backend, "warpx"):
                self._backend.warpx.regrid()

    # ------------------------------------------------------------------
    def tagging_stats(self) -> Dict[str, int]:
        if self._last_mask is None:
            return {"tagged_cells": 0}
        return {"tagged_cells": int(np.sum(self._last_mask))}
