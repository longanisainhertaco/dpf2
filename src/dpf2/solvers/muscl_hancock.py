"""MUSCL-Hancock scheme for MHD."""
from __future__ import annotations

import numpy as np


class MUSCLHancock:
    """Second-order MUSCL-Hancock scheme with HLL Riemann solver."""

    def __init__(self, limiter: str = "minmod") -> None:
        self.limiter = limiter

    def reconstruct(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Piecewise linear reconstruction (placeholder)."""
        return U, U

    def evolve_half_step(self, U_L: np.ndarray, U_R: np.ndarray, dt: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
        """Half-step evolution (placeholder)."""
        return U_L, U_R

    def compute_fluxes(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """HLL Riemann solver (placeholder)."""
        return 0.5 * (U_L + U_R)
