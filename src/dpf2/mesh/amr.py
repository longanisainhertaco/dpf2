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

    shape: tuple[int, int, int]
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

        mask = np.zeros(self.shape, dtype=bool)
        grad_thresh = self.criteria.get("gradient_threshold")
        if density is not None and grad_thresh is not None:
            mask |= plasma_gradient_refinement(np.asarray(density), grad_thresh)

        wave_thresh = self.criteria.get("wavefront_threshold")
        if field is not None and prev_field is not None and wave_thresh is not None:
            mask |= wavefront_refinement(np.asarray(field), np.asarray(prev_field), wave_thresh)

        self._last_mask = mask

        if self._backend and hasattr(self._backend, "amr"):
            self._backend.amr.tag_cells(mask.astype(np.int8))
            if hasattr(self._backend, "warpx"):
                self._backend.warpx.regrid()

    # ------------------------------------------------------------------
    def tagging_stats(self) -> Dict[str, int]:
        if self._last_mask is None:
            return {"tagged_cells": 0}
        return {"tagged_cells": int(self._last_mask.sum())}
