from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import logging
import numpy as np

try:  # pragma: no cover - optional dependency
    import pyamrex  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade if backend missing
    pyamrex = None  # type: ignore

logger = logging.getLogger(__name__)


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
    arr = np.asarray(lambda_D)
    return arr < threshold


def ion_inertial_length_refinement(d_i: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where the ion inertial length ``d_i`` falls below ``threshold``."""
    arr = np.asarray(d_i)
    return arr < threshold


def pressure_gradient_refinement(pressure: np.ndarray, threshold: float) -> np.ndarray:
    """Convenience wrapper applying :func:`plasma_gradient_refinement` to pressure."""
    return plasma_gradient_refinement(pressure, threshold)


def current_density_refinement(current: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where the current density magnitude exceeds ``threshold``."""
    arr = np.asarray(current)
    if arr.shape[-1] != 3:
        raise ValueError("current vectors must have three components")
    mag = np.sqrt(np.sum(arr**2, axis=-1))
    return mag > threshold


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
    ) -> Dict[str, int]:
        """Apply refinement criteria and notify the backend.

        Parameters
        ----------
        plasma_state:
            Mapping of field names to arrays.  Recognised keys include
            ``density``, ``field``, ``lambda_D``, ``d_i``, ``pressure`` and
            ``current``.
        prev_field:
            Optional previous field for wavefront detection.

        Returns
        -------
        dict
            Statistics about tagged cells for each criterion along with the
            total number of tagged cells.
        """
        if plasma_state is None:
            return {}

        density = plasma_state.get("density")
        field = plasma_state.get("field")
        lambda_D = plasma_state.get("lambda_D")
        d_i = plasma_state.get("d_i")
        pressure = plasma_state.get("pressure")
        current = plasma_state.get("current")

        mask = np.zeros(self.shape, dtype=bool)
        stats: Dict[str, int] = {}

        grad_thresh = self.criteria.get("gradient_threshold")
        if density is not None and grad_thresh is not None:
            gmask = plasma_gradient_refinement(density, grad_thresh)
            stats["gradient"] = int(np.sum(gmask))
            mask |= gmask
            if stats["gradient"]:
                logger.info("AMR gradient trigger tagged %d cells", stats["gradient"])

        wave_thresh = self.criteria.get("wavefront_threshold")
        if field is not None and prev_field is not None and wave_thresh is not None:
            wmask = wavefront_refinement(field, prev_field, wave_thresh)
            stats["wavefront"] = int(np.sum(wmask))
            mask |= wmask
            if stats["wavefront"]:
                logger.info("AMR wavefront trigger tagged %d cells", stats["wavefront"])

        ld_thresh = self.criteria.get("lambda_D_threshold")
        if lambda_D is not None and ld_thresh is not None:
            ldmask = debye_length_refinement(lambda_D, ld_thresh)
            stats["lambda_D"] = int(np.sum(ldmask))
            mask |= ldmask
            if stats["lambda_D"]:
                logger.info("AMR λ_D trigger tagged %d cells", stats["lambda_D"])

        di_thresh = self.criteria.get("d_i_threshold")
        if d_i is not None and di_thresh is not None:
            dimask = ion_inertial_length_refinement(d_i, di_thresh)
            stats["d_i"] = int(np.sum(dimask))
            mask |= dimask
            if stats["d_i"]:
                logger.info("AMR d_i trigger tagged %d cells", stats["d_i"])

        gradp_thresh = self.criteria.get("pressure_gradient_threshold")
        if pressure is not None and gradp_thresh is not None:
            pmask = pressure_gradient_refinement(pressure, gradp_thresh)
            stats["pressure_gradient"] = int(np.sum(pmask))
            mask |= pmask
            if stats["pressure_gradient"]:
                logger.info(
                    "AMR |∇p| trigger tagged %d cells", stats["pressure_gradient"]
                )

        J_thresh = self.criteria.get("current_density_threshold")
        if current is not None and J_thresh is not None:
            jmask = current_density_refinement(current, J_thresh)
            stats["current_density"] = int(np.sum(jmask))
            mask |= jmask
            if stats["current_density"]:
                logger.info(
                    "AMR |J| trigger tagged %d cells", stats["current_density"]
                )

        self._last_mask = np.array(mask)
        stats["tagged_cells"] = int(np.sum(self._last_mask))

        if self._backend and hasattr(self._backend, "amr"):
            self._backend.amr.tag_cells(self._last_mask.astype(np.int8))
            if hasattr(self._backend, "warpx"):
                self._backend.warpx.regrid()

        return stats

    # ------------------------------------------------------------------
    def tagging_stats(self) -> Dict[str, int]:
        if self._last_mask is None:
            return {"tagged_cells": 0}
        return {"tagged_cells": int(np.sum(self._last_mask))}
