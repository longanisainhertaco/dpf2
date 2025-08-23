"""Skeleton Hall-MHD solver with placeholders for advanced physics.

This module defines data structures and a minimal solver interface for a
future 3-D resistive Hall-MHD implementation with anisotropic transport,
constrained transport (CT) for divergence-free magnetic fields, and
hooks for AMR integration.  The solver is intentionally incomplete but
provides typed containers and method stubs so that further development
can proceed incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import PlasmaSolverBase

__all__ = ["MHDState", "HallMHDSolver"]


def _dd(f: np.ndarray, axis: int) -> np.ndarray:
    """Centered difference with periodic boundaries and unit spacing."""
    return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) * 0.5


def _divergence(vec: np.ndarray) -> np.ndarray:
    """Compute the discrete divergence of a vector field."""
    return _dd(vec[..., 0], 0) + _dd(vec[..., 1], 1) + _dd(vec[..., 2], 2)


def _curl(vec: np.ndarray) -> np.ndarray:
    """Compute the discrete curl of a vector field."""
    cx = _dd(vec[..., 2], 1) - _dd(vec[..., 1], 2)
    cy = _dd(vec[..., 0], 2) - _dd(vec[..., 2], 0)
    cz = _dd(vec[..., 1], 0) - _dd(vec[..., 0], 1)
    return np.stack((cx, cy, cz), axis=-1)


def _project_div_free(B: np.ndarray) -> np.ndarray:
    """Project a magnetic field onto its divergence-free component."""
    nx, ny, nz, _ = B.shape
    B_hat = np.fft.fftn(B, axes=(0, 1, 2))
    kx = 2 * np.pi * np.fft.fftfreq(nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny)
    kz = 2 * np.pi * np.fft.fftfreq(nz)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0  # avoid divide-by-zero for mean mode
    k_dot_B = kx * B_hat[..., 0] + ky * B_hat[..., 1] + kz * B_hat[..., 2]
    for i, k in enumerate((kx, ky, kz)):
        B_hat[..., i] -= k * k_dot_B / k2
    return np.fft.ifftn(B_hat, axes=(0, 1, 2)).real


@dataclass
class MHDState:
    """State container for the MHD variables.

    Attributes
    ----------
    rho : ndarray
        Mass density [kg/m^3].
    mom : ndarray
        Momentum density vector [kg/m^2/s].
    energy : ndarray
        Total energy density [J/m^3].
    B : ndarray
        Magnetic field vector [T].
    Te : ndarray | None
        Electron temperature [K] when using two-temperature models.
    Ti : ndarray | None
        Ion temperature [K] when using two-temperature models.
    """

    rho: np.ndarray
    mom: np.ndarray
    energy: np.ndarray
    B: np.ndarray
    Te: np.ndarray | None = None
    Ti: np.ndarray | None = None


@dataclass
class HallMHDSolver(PlasmaSolverBase):
    """Stub for a 3-D Hall-MHD solver with CT and AMR hooks.

    Parameters
    ----------
    mesh : Any
        Placeholder for mesh/AMR hierarchy object.
    """

    mesh: Any = field(default=None)
    eta: float = 0.0
    hall_coeff: float = 0.0
    rad_coeff: float = 0.0

    def step(self, state: MHDState, dt: float) -> MHDState:  # pragma: no cover - skeleton
        """Advance the state by ``dt`` seconds using a simplified MHD update."""

        gamma = 5.0 / 3.0

        rho = state.rho.copy()
        mom = state.mom.copy()
        energy = state.energy.copy()
        B = state.B.copy()

        v = mom / rho[..., None]
        B2 = np.sum(B**2, axis=-1)
        kinetic = 0.5 * rho * np.sum(v**2, axis=-1)
        magnetic = 0.5 * B2
        p = (gamma - 1.0) * (energy - kinetic - magnetic)

        # --- Flux computation (Lax-Friedrichs style) ---
        flux_rho = np.zeros((3,) + rho.shape)
        flux_mom = np.zeros((3,) + mom.shape)
        flux_energy = np.zeros((3,) + energy.shape)
        vdotB = np.sum(v * B, axis=-1)

        for i in range(3):
            flux_rho[i] = rho * v[..., i]
            for j in range(3):
                flux_mom[i][..., j] = mom[..., j] * v[..., i]
                if i == j:
                    flux_mom[i][..., j] += p + magnetic
                flux_mom[i][..., j] -= B[..., i] * B[..., j]
            flux_energy[i] = (energy + p + magnetic) * v[..., i] - vdotB * B[..., i]

        def div_flux(F):
            return _dd(F[0], 0) + _dd(F[1], 1) + _dd(F[2], 2)

        rho -= dt * div_flux(flux_rho)
        new_mom = np.empty_like(mom)
        for j in range(3):
            new_mom[..., j] = mom[..., j] - dt * div_flux(flux_mom[:, ..., j])
        mom = new_mom
        energy -= dt * div_flux(flux_energy)

        # --- Constrained transport via electric fields ---
        J = _curl(B)
        E = -np.cross(v, B) + self.eta * J
        if self.hall_coeff != 0.0:
            E += self.hall_coeff * np.cross(J, B) / rho[..., None]
        B -= dt * _curl(E)
        B = _project_div_free(B)

        # --- Source terms ---
        if self.eta != 0.0:
            energy += dt * self.eta * np.sum(J**2, axis=-1)
        if self.rad_coeff != 0.0:
            energy -= dt * self.rad_coeff * energy

        return MHDState(
            rho=rho,
            mom=mom,
            energy=energy,
            B=B,
            Te=None if state.Te is None else state.Te.copy(),
            Ti=None if state.Ti is None else state.Ti.copy(),
        )
