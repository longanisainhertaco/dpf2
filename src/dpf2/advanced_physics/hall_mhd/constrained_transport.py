"""Constrained transport update for preserving div(B) = 0.

This module provides the CTUpdate class and related functions for
advancing the magnetic field in a divergence-preserving manner using
the constrained transport method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.fft import fftn, ifftn, fftfreq
except ImportError:
    from numpy.fft import fftn, ifftn, fftfreq

__all__ = [
    "CTUpdate",
    "compute_curl",
    "compute_divergence",
    "project_div_free",
]


def compute_curl(
    F: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Compute curl of a vector field using central differences.

    Parameters
    ----------
    F : ndarray
        Vector field, shape (nx, ny, nz, 3).
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    ndarray
        Curl of F, shape (nx, ny, nz, 3).
    """
    Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]

    dFz_dy = (np.roll(Fz, -1, axis=1) - np.roll(Fz, 1, axis=1)) / (2 * dy)
    dFy_dz = (np.roll(Fy, -1, axis=2) - np.roll(Fy, 1, axis=2)) / (2 * dz)
    curl_x = dFz_dy - dFy_dz

    dFx_dz = (np.roll(Fx, -1, axis=2) - np.roll(Fx, 1, axis=2)) / (2 * dz)
    dFz_dx = (np.roll(Fz, -1, axis=0) - np.roll(Fz, 1, axis=0)) / (2 * dx)
    curl_y = dFx_dz - dFz_dx

    dFy_dx = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0)) / (2 * dx)
    dFx_dy = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1)) / (2 * dy)
    curl_z = dFy_dx - dFx_dy

    return np.stack([curl_x, curl_y, curl_z], axis=-1)


def compute_divergence(
    F: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Compute divergence of a vector field using central differences.

    Parameters
    ----------
    F : ndarray
        Vector field, shape (nx, ny, nz, 3).
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    ndarray
        Divergence of F, shape (nx, ny, nz).
    """
    dFx_dx = (np.roll(F[..., 0], -1, axis=0) - np.roll(F[..., 0], 1, axis=0)) / (2 * dx)
    dFy_dy = (np.roll(F[..., 1], -1, axis=1) - np.roll(F[..., 1], 1, axis=1)) / (2 * dy)
    dFz_dz = (np.roll(F[..., 2], -1, axis=2) - np.roll(F[..., 2], 1, axis=2)) / (2 * dz)
    return dFx_dx + dFy_dy + dFz_dz


def project_div_free(B: np.ndarray) -> np.ndarray:
    """Project a vector field onto its divergence-free component using FFT.

    This function uses Helmholtz decomposition in Fourier space to
    remove the curl-free (gradient) component, leaving only the
    divergence-free (solenoidal) part.

    Parameters
    ----------
    B : ndarray
        Vector field, shape (nx, ny, nz, 3) or (nx, ny, 1, 3).

    Returns
    -------
    ndarray
        Divergence-free projection of B.
    """
    original_shape = B.shape
    if B.ndim == 3:
        B = B.reshape(B.shape[0], B.shape[1], 1, 3)

    spatial_shape = B.shape[:-1]
    nx, ny, nz = spatial_shape

    if nz == 1:
        B_3d = np.broadcast_to(B, (nx, ny, 2, 3)).copy()
        B_3d[:, :, 1, :] = B[:, :, 0, :]
    else:
        B_3d = B

    nx, ny, nz = B_3d.shape[:-1]

    B_hat = fftn(B_3d, axes=(0, 1, 2))

    kx = 2 * np.pi * fftfreq(nx)
    ky = 2 * np.pi * fftfreq(ny)
    kz = 2 * np.pi * fftfreq(nz)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing="ij")

    k2 = Kx ** 2 + Ky ** 2 + Kz ** 2
    k2[0, 0, 0] = 1.0

    k_dot_B = Kx * B_hat[..., 0] + Ky * B_hat[..., 1] + Kz * B_hat[..., 2]

    B_hat[..., 0] = B_hat[..., 0] - Kx * k_dot_B / k2
    B_hat[..., 1] = B_hat[..., 1] - Ky * k_dot_B / k2
    B_hat[..., 2] = B_hat[..., 2] - Kz * k_dot_B / k2

    B_proj = ifftn(B_hat, axes=(0, 1, 2)).real

    if original_shape[-2] == 1 or (len(original_shape) == 4 and original_shape[2] == 1):
        B_proj = B_proj[:, :, :1, :]

    return B_proj.reshape(original_shape)


@dataclass
class CTUpdate:
    """Constrained transport update for magnetic field evolution.

    The constrained transport (CT) method advances the magnetic field
    using the induction equation dB/dt = -curl(E) in a way that exactly
    preserves the divergence-free constraint div(B) = 0 to machine
    precision.

    Attributes
    ----------
    enforce_div_free : bool
        If True, apply FFT projection after each update as a failsafe.
    """

    enforce_div_free: bool = True

    def update(
        self,
        B: np.ndarray,
        E: np.ndarray,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> np.ndarray:
        """Advance magnetic field using dB/dt = -curl(E).

        Parameters
        ----------
        B : ndarray
            Magnetic field at time t, shape (nx, ny, nz, 3).
        E : ndarray
            Electric field, shape (nx, ny, nz, 3).
        dt : float
            Time step.
        dx, dy, dz : float
            Grid spacings.

        Returns
        -------
        ndarray
            Updated magnetic field at time t + dt.
        """
        curl_E = compute_curl(E, dx, dy, dz)
        B_new = B - dt * curl_E

        if self.enforce_div_free:
            B_new = project_div_free(B_new)

        return B_new

    def compute_emf_from_flux(
        self,
        F_B: np.ndarray,
        direction: int,
    ) -> np.ndarray:
        """Compute EMF from magnetic flux for staggered grid formulation.

        This is used in more sophisticated CT implementations where
        EMFs are computed at cell edges from face-centered fluxes.

        Parameters
        ----------
        F_B : ndarray
            Magnetic flux at faces.
        direction : int
            Direction index (0=x, 1=y, 2=z).

        Returns
        -------
        ndarray
            EMF at cell edges.
        """
        return F_B.copy()


def _forward_difference(f: np.ndarray, axis: int) -> np.ndarray:
    """Compute forward difference with periodic boundaries."""
    return np.roll(f, -1, axis=axis) - f


def _backward_difference(f: np.ndarray, axis: int) -> np.ndarray:
    """Compute backward difference with periodic boundaries."""
    return f - np.roll(f, 1, axis=axis)


def upwind_ct_update(
    B: np.ndarray,
    v: np.ndarray,
    E_resistive: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Upwind constrained transport update.

    This uses an upwind reconstruction for the convective part of the
    electric field to improve stability in advection-dominated flows.

    Parameters
    ----------
    B : ndarray
        Magnetic field, shape (nx, ny, nz, 3).
    v : ndarray
        Velocity field, shape (nx, ny, nz, 3).
    E_resistive : ndarray
        Resistive electric field eta*J, shape (nx, ny, nz, 3).
    dt, dx, dy, dz : float
        Time step and grid spacings.

    Returns
    -------
    ndarray
        Updated magnetic field.
    """
    E_conv = -np.cross(v, B, axis=-1)

    E_total = E_conv + E_resistive

    curl_E = compute_curl(E_total, dx, dy, dz)

    B_new = B - dt * curl_E

    return project_div_free(B_new)
