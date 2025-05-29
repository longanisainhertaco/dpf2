"""Resistive MHD equations in 2D cylindrical geometry."""
from __future__ import annotations

import numpy as np


class ResistiveMHD:
    """Minimal representation of resistive MHD equations."""

    def __init__(self, gamma: float = 5 / 3) -> None:
        self.gamma = gamma
        self.equations = [
            "density",
            "momentum_r",
            "momentum_z",
            "energy",
            "B_r",
            "B_z",
            "B_phi",
        ]

    def conservative_variables(self, primitives: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative form."""
        # Placeholder conversion
        return primitives

    def flux_function(self, U: np.ndarray, direction: str) -> np.ndarray:
        """Compute MHD fluxes."""
        # Placeholder flux calculation
        return np.zeros_like(U)

    def source_terms(self, U: np.ndarray) -> np.ndarray:
        """Return resistive and geometric source terms."""
        return np.zeros_like(U)
