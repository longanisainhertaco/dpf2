from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import logging
import numpy as np

from .criteria import (
    plasma_gradient_refinement,
    debye_length_refinement,
    ion_inertial_length_refinement,
    ion_skin_depth_refinement,
    pressure_gradient_refinement,
    current_density_refinement,
    current_gradient_refinement,
    wavefront_refinement,
)

try:  # pragma: no cover - optional dependency
    import pyamrex  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade if backend missing
    pyamrex = None  # type: ignore

logger = logging.getLogger(__name__)


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
        """Apply refinement criteria and notify the backend."""
        if plasma_state is None:
            return {}

        density = plasma_state.get("density")
        field = plasma_state.get("field")
        lambda_D = plasma_state.get("lambda_D")
        d_i = plasma_state.get("d_i")
        pressure = plasma_state.get("pressure")
        current = plasma_state.get("current")

        def _zeros_bool(shape):
            if len(shape) == 1:
                return [False] * shape[0]
            return [_zeros_bool(shape[1:]) for _ in range(shape[0])]

        def _or_inplace(target, src):
            if isinstance(target[0], list):
                for t, s in zip(target, getattr(src, "data", src)):
                    _or_inplace(t, s)
            else:
                for i in range(len(target)):
                    target[i] = target[i] or bool(getattr(src, "data", src)[i])

        mask = _zeros_bool(self.shape)
        stats: Dict[str, int] = {}

        grad_thresh = self.criteria.get("gradient_threshold")
        if density is not None and grad_thresh is not None:
            gmask = plasma_gradient_refinement(density, grad_thresh)
            stats["gradient"] = int(np.sum(getattr(gmask, "data", gmask)))
            _or_inplace(mask, gmask)
            if stats["gradient"]:
                logger.info("AMR gradient trigger tagged %d cells", stats["gradient"])

        wave_thresh = self.criteria.get("wavefront_threshold")
        if field is not None and prev_field is not None and wave_thresh is not None:
            wmask = wavefront_refinement(field, prev_field, wave_thresh)
            stats["wavefront"] = int(np.sum(getattr(wmask, "data", wmask)))
            _or_inplace(mask, wmask)
            if stats["wavefront"]:
                logger.info("AMR wavefront trigger tagged %d cells", stats["wavefront"])

        ld_thresh = self.criteria.get("lambda_D_threshold")
        if lambda_D is not None and ld_thresh is not None:
            ldmask = debye_length_refinement(lambda_D, ld_thresh)
            stats["lambda_D"] = int(np.sum(getattr(ldmask, "data", ldmask)))
            _or_inplace(mask, ldmask)
            if stats["lambda_D"]:
                logger.info("AMR λ_D trigger tagged %d cells", stats["lambda_D"])

        di_thresh = self.criteria.get("d_i_threshold")
        if d_i is not None and di_thresh is not None:
            dimask = ion_inertial_length_refinement(d_i, di_thresh)
            stats["d_i"] = int(np.sum(getattr(dimask, "data", dimask)))
            _or_inplace(mask, dimask)
            if stats["d_i"]:
                logger.info("AMR d_i trigger tagged %d cells", stats["d_i"])

        gradp_thresh = self.criteria.get("pressure_gradient_threshold")
        if pressure is not None and gradp_thresh is not None:
            pmask = pressure_gradient_refinement(pressure, gradp_thresh)
            stats["pressure_gradient"] = int(np.sum(getattr(pmask, "data", pmask)))
            _or_inplace(mask, pmask)
            if stats["pressure_gradient"]:
                logger.info(
                    "AMR |∇p| trigger tagged %d cells", stats["pressure_gradient"]
                )

        J_thresh = self.criteria.get("current_density_threshold")
        if current is not None and J_thresh is not None:
            jmask = current_density_refinement(current, J_thresh)
            stats["current_density"] = int(np.sum(getattr(jmask, "data", jmask)))
            _or_inplace(mask, jmask)
            if stats["current_density"]:
                logger.info(
                    "AMR |J| trigger tagged %d cells", stats["current_density"]
                )

        Jgrad_thresh = self.criteria.get("current_gradient_threshold")
        if current is not None and Jgrad_thresh is not None:
            cgmask = current_gradient_refinement(current, Jgrad_thresh)
            stats["current_gradient"] = int(np.sum(getattr(cgmask, "data", cgmask)))
            _or_inplace(mask, cgmask)
            if stats["current_gradient"]:
                logger.info(
                    "AMR ∇|J| trigger tagged %d cells", stats["current_gradient"]
                )

        def _count(arr):
            if isinstance(arr, list):
                return sum(_count(a) for a in arr)
            return int(bool(arr))

        self._last_mask = np.array(mask)
        stats["tagged_cells"] = _count(mask)

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


__all__ = [
    "AMRMesh",
    "plasma_gradient_refinement",
    "debye_length_refinement",
    "ion_inertial_length_refinement",
    "ion_skin_depth_refinement",
    "pressure_gradient_refinement",
    "current_density_refinement",
    "current_gradient_refinement",
    "wavefront_refinement",
]
