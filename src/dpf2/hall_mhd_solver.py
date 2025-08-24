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
from typing import Any, Callable

import numpy as np
from scipy.constants import mu_0

from dpf2.core.bases import PlasmaSolverBase
from .eos import EOSBase, IdealGasEOS
from .chemistry import ChemistryModel, SahaEquilibrium
from .radiation import RadiationBase

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


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod limiter used for MUSCL reconstruction."""
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


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

    The solver now uses a second-order Godunov (MUSCL-Hancock)
    update with a constrained-transport magnetic-field advance and
    an optional Hall term.  Boundary-condition and AMR refinement
    hooks allow external customization of the update.

    Parameters
    ----------
    mesh : Any
        Placeholder for mesh/AMR hierarchy object.
    """

    mesh: Any = field(default=None)
    eos: EOSBase = field(default_factory=IdealGasEOS)
    chemistry: ChemistryModel = field(default_factory=SahaEquilibrium)
    radiation: RadiationBase | None = None
    eta: float = 0.0
    hall_coeff: float = 0.0
    rad_coeff: float = 0.0
    nu_par: float = 0.0
    kappa_par: float = 0.0
    bc: Callable[[MHDState], None] | None = None
    refine: Callable[[MHDState], None] | None = None
    last_pressure: np.ndarray | None = field(init=False, default=None)
    last_ionization: np.ndarray | None = field(init=False, default=None)
    last_rad_loss: np.ndarray | None = field(init=False, default=None)

    def apply_boundary_conditions(self, state: MHDState) -> None:
        """Invoke the boundary-condition hook if provided."""
        if self.bc is not None:
            self.bc(state)

    def amr_refinement(self, state: MHDState) -> None:
        """Invoke the refinement callback if provided."""
        if self.refine is not None:
            self.refine(state)

    def step(self, state: MHDState, dt: float) -> MHDState:  # pragma: no cover - skeleton
        """Advance the state by ``dt`` seconds using a higher-order MHD update.

        The method employs a MUSCL-Hancock Godunov scheme with a
        Rusanov solver for the fluid variables and a constrained
        transport update for the magnetic field.  Hall physics enters
        through the electric field used in the CT step.  Optional
        Braginskii transport terms act along the magnetic-field
        direction.
        """

        self.apply_boundary_conditions(state)

        rho = state.rho.copy()
        mom = state.mom.copy()
        energy = state.energy.copy()
        B = state.B.copy()

        v = mom / rho[..., None]
        B2 = np.sum(B**2, axis=-1)
        kinetic = 0.5 * rho * np.sum(v**2, axis=-1)
        magnetic = 0.5 * B2
        e_internal = energy - kinetic - magnetic
        specific_e = e_internal / rho
        T = self.eos.temperature(rho, specific_e)
        p = self.eos.pressure(rho, T)
        zbar = self.chemistry.ionization_state(rho, T)
        if self.radiation is not None:
            rad_loss = self.radiation.loss(rho, T * zbar)
            energy -= dt * rad_loss
            self.last_rad_loss = rad_loss
        else:
            self.last_rad_loss = None
        self.last_pressure = p
        self.last_ionization = zbar

        # --- High-order Godunov fluxes (MUSCL-Hancock) ---
        gamma = getattr(self.eos, "gamma", 5.0 / 3.0)
        flux_rho = np.zeros((3,) + rho.shape)
        flux_mom = np.zeros((3,) + mom.shape)
        flux_energy = np.zeros((3,) + energy.shape)

        prim_vars = [rho, v[..., 0], v[..., 1], v[..., 2], B[..., 0], B[..., 1], B[..., 2], p]
        for i in range(3):
            slopes = [
                _minmod(var - np.roll(var, 1, axis=i), np.roll(var, -1, axis=i) - var)
                for var in prim_vars
            ]
            left_states = [var - 0.5 * s for var, s in zip(prim_vars, slopes)]
            right_states = [
                np.roll(var, -1, axis=i) + 0.5 * np.roll(s, -1, axis=i)
                for var, s in zip(prim_vars, slopes)
            ]

            rho_L, vx_L, vy_L, vz_L, Bx_L, By_L, Bz_L, p_L = [ls for ls in left_states]
            rho_R, vx_R, vy_R, vz_R, Bx_R, By_R, Bz_R, p_R = [rs for rs in right_states]
            v_L = np.stack((vx_L, vy_L, vz_L), axis=-1)
            v_R = np.stack((vx_R, vy_R, vz_R), axis=-1)
            B_L = np.stack((Bx_L, By_L, Bz_L), axis=-1)
            B_R = np.stack((Bx_R, By_R, Bz_R), axis=-1)

            B2_L = np.sum(B_L**2, axis=-1)
            B2_R = np.sum(B_R**2, axis=-1)
            kinetic_L = 0.5 * rho_L * np.sum(v_L**2, axis=-1)
            kinetic_R = 0.5 * rho_R * np.sum(v_R**2, axis=-1)
            energy_L = p_L / (gamma - 1.0) + kinetic_L + 0.5 * B2_L
            energy_R = p_R / (gamma - 1.0) + kinetic_R + 0.5 * B2_R

            mom_L = rho_L[..., None] * v_L
            mom_R = rho_R[..., None] * v_R

            vdotB_L = np.sum(v_L * B_L, axis=-1)
            vdotB_R = np.sum(v_R * B_R, axis=-1)

            F_rho_L = rho_L * v_L[..., i]
            F_rho_R = rho_R * v_R[..., i]

            F_mom_L = np.zeros_like(mom_L)
            F_mom_R = np.zeros_like(mom_R)
            for j in range(3):
                F_mom_L[..., j] = mom_L[..., j] * v_L[..., i]
                F_mom_R[..., j] = mom_R[..., j] * v_R[..., i]
                if i == j:
                    F_mom_L[..., j] += p_L + 0.5 * B2_L
                    F_mom_R[..., j] += p_R + 0.5 * B2_R
                F_mom_L[..., j] -= B_L[..., i] * B_L[..., j]
                F_mom_R[..., j] -= B_R[..., i] * B_R[..., j]

            F_energy_L = (energy_L + p_L + 0.5 * B2_L) * v_L[..., i] - vdotB_L * B_L[..., i]
            F_energy_R = (energy_R + p_R + 0.5 * B2_R) * v_R[..., i] - vdotB_R * B_R[..., i]

            cs_L = np.sqrt(gamma * p_L / rho_L)
            cs_R = np.sqrt(gamma * p_R / rho_R)
            ca_L = np.sqrt(B2_L / rho_L)
            ca_R = np.sqrt(B2_R / rho_R)
            cfast_L = np.sqrt(cs_L**2 + ca_L**2)
            cfast_R = np.sqrt(cs_R**2 + ca_R**2)
            SL = np.minimum(v_L[..., i] - cfast_L, v_R[..., i] - cfast_R)
            SR = np.maximum(v_L[..., i] + cfast_L, v_R[..., i] + cfast_R)

            U_L = np.stack((rho_L, mom_L[..., 0], mom_L[..., 1], mom_L[..., 2], energy_L), axis=-1)
            U_R = np.stack((rho_R, mom_R[..., 0], mom_R[..., 1], mom_R[..., 2], energy_R), axis=-1)
            F_L = np.stack((F_rho_L, F_mom_L[..., 0], F_mom_L[..., 1], F_mom_L[..., 2], F_energy_L), axis=-1)
            F_R = np.stack((F_rho_R, F_mom_R[..., 0], F_mom_R[..., 1], F_mom_R[..., 2], F_energy_R), axis=-1)

            denom = SR - SL
            denom[denom == 0] = 1e-30
            flux = np.where(
                (SL[..., None] >= 0),
                F_L,
                np.where(
                    (SR[..., None] <= 0),
                    F_R,
                    (SR[..., None] * F_L - SL[..., None] * F_R + SL[..., None] * SR[..., None] * (U_R - U_L))
                    / denom[..., None],
                ),
            )

            flux_rho[i] = flux[..., 0]
            flux_mom[i] = flux[..., 1:4]
            flux_energy[i] = flux[..., 4]

        for i in range(3):
            rho -= dt * (flux_rho[i] - np.roll(flux_rho[i], 1, axis=i))
            energy -= dt * (flux_energy[i] - np.roll(flux_energy[i], 1, axis=i))
            for j in range(3):
                mom[..., j] -= dt * (flux_mom[i][..., j] - np.roll(flux_mom[i][..., j], 1, axis=i))

        # --- Constrained transport via electric fields ---
        J = _curl(B)
        E = -np.cross(v, B) + self.eta * J
        if self.hall_coeff != 0.0:
            ne = rho * np.maximum(zbar, 1e-30)
            E += self.hall_coeff * np.cross(J, B) / ne[..., None]
        B -= dt * _curl(E)
        B = _project_div_free(B)

        # --- Braginskii viscosity (parallel component) ---
        if self.nu_par != 0.0:
            b = B / np.sqrt(B2 + 1e-30)[..., None]
            for comp in range(3):
                grad_par = sum(b[..., i] * _dd(v[..., comp], i) for i in range(3))
                visc_flux = self.nu_par * b * grad_par[..., None]
                mom[..., comp] += dt * (
                    _dd(visc_flux[..., 0], 0)
                    + _dd(visc_flux[..., 1], 1)
                    + _dd(visc_flux[..., 2], 2)
                )
                energy += dt * self.nu_par * grad_par**2 * rho

        # --- Braginskii thermal conduction (parallel) ---
        if self.kappa_par != 0.0:
            T = p / rho
            b = B / np.sqrt(B2 + 1e-30)[..., None]
            gradT_par = sum(b[..., i] * _dd(T, i) for i in range(3))
            q = -self.kappa_par * b * gradT_par[..., None]
            energy -= dt * (
                _dd(q[..., 0], 0) + _dd(q[..., 1], 1) + _dd(q[..., 2], 2)
            )

        # --- Source terms ---
        if self.eta != 0.0:
            energy += dt * self.eta * np.sum(J**2, axis=-1)

        new_state = MHDState(
            rho=rho,
            mom=mom,
            energy=energy,
            B=B,
            Te=None if state.Te is None else state.Te.copy(),
            Ti=None if state.Ti is None else state.Ti.copy(),
        )

        self.apply_boundary_conditions(new_state)
        self.amr_refinement(new_state)

        return new_state

    def compute_plasma_inductance(self, state: MHDState, current: float, cell_volume: float = 1.0) -> float:
        """Estimate plasma inductance from magnetic energy.

        Parameters
        ----------
        state : MHDState
            Current plasma state.
        current : float
            Circuit current in amperes.
        cell_volume : float, optional
            Volume of a single cell, by default 1.0.

        Returns
        -------
        float
            Estimated inductance in henries.
        """
        B2 = np.sum(state.B ** 2)
        magnetic_energy = B2 * cell_volume / (2 * mu_0)
        if current == 0.0:
            return 0.0
        return 2 * magnetic_energy / (current ** 2)
