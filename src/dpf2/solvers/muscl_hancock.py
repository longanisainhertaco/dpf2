"""MUSCL-Hancock scheme for MHD."""
from __future__ import annotations

import numpy as np


class MUSCLHancock:
    """Second-order MUSCL-Hancock scheme with HLL Riemann solver.

    The implementation here supports the 1D compressible Euler equations for
    an ideal gas.  It provides the building blocks of the MUSCL–Hancock
    finite-volume scheme: slope-limited reconstruction, a predictor step that
    evolves the reconstructed states to half a time step, and an Harten–Lax–
    van Leer (HLL) Riemann solver used to compute interface fluxes.
    """

    gamma: float = 1.4

    def __init__(self, limiter: str = "minmod") -> None:  # noqa: D401 - short
        self.limiter = limiter

    # ------------------------------------------------------------------
    # reconstruction utilities
    # ------------------------------------------------------------------
    def _minmod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

    def _slope(self, U: np.ndarray) -> np.ndarray:
        """Return slope-limited gradients using the configured limiter."""
        d_minus = U[1:-1] - U[:-2]
        d_plus = U[2:] - U[1:-1]

        if self.limiter == "minmod":
            sig = self._minmod(d_minus, d_plus)
        else:  # fallback to minmod for unknown limiters
            sig = self._minmod(d_minus, d_plus)

        # Pad with zeros at boundaries (first order there)
        sig = np.vstack([np.zeros_like(sig[0:1]), sig, np.zeros_like(sig[0:1])])
        return sig

    def reconstruct(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Piecewise linear reconstruction with slope limiting.

        Parameters
        ----------
        U : ndarray (N, nvar)
            Cell averaged conservative variables.

        Returns
        -------
        tuple of ndarray
            Left and right states within each cell.
        """
        slope = 0.5 * self._slope(U)
        U_L = U - slope
        U_R = U + slope
        return U_L, U_R

    # ------------------------------------------------------------------
    # predictor step
    # ------------------------------------------------------------------
    def _flux(self, U: np.ndarray) -> np.ndarray:
        rho, u, p = self._primitive(U)
        E = U[:, 2]
        return np.stack(
            [rho * u, rho * u**2 + p, u * (E + p)],
            axis=1,
        )

    def _primitive(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = U[:, 0]
        u = U[:, 1] / rho
        E = U[:, 2]
        p = (self.gamma - 1.0) * (E - 0.5 * rho * u**2)
        return rho, u, p

    def evolve_half_step(
        self, U_L: np.ndarray, U_R: np.ndarray, dt: float, dx: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict left/right states at interfaces to half a time step.

        ``U_L`` and ``U_R`` are the reconstructed left and right states within
        each cell returned by :meth:`reconstruct`.  They have shape ``(N,nvar)``.
        The function returns the left and right states at cell interfaces after
        evolving them for ``dt/2``.
        """
        F_L = self._flux(U_L)
        F_R = self._flux(U_R)
        dU = -(F_R - F_L) / dx

        U_L_half = U_L + 0.5 * dt * dU
        U_R_half = U_R + 0.5 * dt * dU

        # Interface states: left side of cell i is right side of cell i-1
        U_L_int = np.zeros((U_L.shape[0] + 1, U_L.shape[1]))
        U_R_int = np.zeros_like(U_L_int)
        U_L_int[1:] = U_R_half
        U_R_int[:-1] = U_L_half
        # copy boundary states
        U_L_int[0] = U_L_half[0]
        U_R_int[0] = U_L_half[0]
        U_L_int[-1] = U_R_half[-1]
        U_R_int[-1] = U_R_half[-1]
        return U_L_int, U_R_int

    # ------------------------------------------------------------------
    # HLL Riemann solver
    # ------------------------------------------------------------------
    def compute_fluxes(self, U_L: np.ndarray, U_R: np.ndarray) -> np.ndarray:
        """Compute interface fluxes using the HLL Riemann solver."""

        rho_L, u_L, p_L = self._primitive(U_L)
        rho_R, u_R, p_R = self._primitive(U_R)

        E_L = U_L[:, 2]
        E_R = U_R[:, 2]

        F_L = np.stack([rho_L * u_L, rho_L * u_L**2 + p_L, u_L * (E_L + p_L)], axis=1)
        F_R = np.stack([rho_R * u_R, rho_R * u_R**2 + p_R, u_R * (E_R + p_R)], axis=1)

        c_L = np.sqrt(self.gamma * p_L / rho_L)
        c_R = np.sqrt(self.gamma * p_R / rho_R)

        S_L = np.minimum(u_L - c_L, u_R - c_R)
        S_R = np.maximum(u_L + c_L, u_R + c_R)

        flux = np.where(
            (S_L[:, None] >= 0),
            F_L,
            np.where(
                (S_R[:, None] <= 0),
                F_R,
                (
                    (S_R[:, None] * F_L - S_L[:, None] * F_R + S_L[:, None] * S_R[:, None] * (U_R - U_L))
                    / (S_R - S_L)[:, None]
                ),
            ),
        )
        return flux
