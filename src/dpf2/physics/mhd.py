"""Resistive MHD equations in 2D cylindrical geometry."""
from __future__ import annotations

import numpy as np


class ResistiveMHD:
    """Minimal representation of resistive MHD equations.

    Extended with an anisotropic conductivity tensor and a simple Hall term to
    demonstrate cross-field heat conduction and Hall-induced currents.
    """

    def __init__(
        self,
        gamma: float = 5 / 3,
        sigma_parallel: float = 1.0,
        sigma_perp: float = 1.0,
        hall_param: float = 0.0,
    ) -> None:
        self.gamma = gamma
        self.sigma_parallel = sigma_parallel
        self.sigma_perp = sigma_perp
        self.hall_param = hall_param
        self.equations = [
            "density",
            "momentum_r",
            "momentum_z",
            "energy",
            "B_r",
            "B_z",
            "B_phi",
        ]

    def conductivity_tensor(self) -> np.ndarray:
        """Return the anisotropic conductivity tensor."""
        return np.diag([self.sigma_perp, self.sigma_perp, self.sigma_parallel])

    def cross_field_conduction(self, grad_T: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute heat flux with anisotropic conductivity."""
        b = B / (np.linalg.norm(B) + 1e-12)
        grad_par = np.dot(grad_T, b) * b
        grad_perp = grad_T - grad_par
        return -self.sigma_parallel * grad_par - self.sigma_perp * grad_perp

    def hall_current(self, J: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Return current including Hall-induced contribution."""
        return J + self.hall_param * np.cross(J, B)

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
