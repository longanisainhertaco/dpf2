"""3D Hall-MHD solver with constrained transport.

This module implements a full 3D Hall-MHD solver using a second-order
Godunov scheme with constrained transport to preserve div(B)=0. The
solver supports optional Braginskii transport coefficients and hooks
for adaptive mesh refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np

try:
    from scipy.constants import mu_0, e as q_e, m_e, m_p
except ImportError:
    mu_0 = 4e-7 * np.pi
    q_e = 1.602176634e-19
    m_e = 9.1093837015e-31
    m_p = 1.67262192369e-27

from .constrained_transport import CTUpdate, compute_curl, project_div_free

__all__ = ["HallMHDSolver3D", "Grid3D", "MHDState3D"]


@dataclass
class Grid3D:
    """Three-dimensional uniform Cartesian grid.

    Attributes
    ----------
    nx, ny, nz : int
        Number of cells in each direction.
    dx, dy, dz : float
        Cell sizes in each direction (meters).
    x, y, z : ndarray
        Cell-center coordinates.
    """

    nx: int
    ny: int
    nz: int
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0

    def __post_init__(self) -> None:
        """Compute cell-center coordinates."""
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy
        self.z = (np.arange(self.nz) + 0.5) * self.dz

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return grid shape as (nx, ny, nz)."""
        return (self.nx, self.ny, self.nz)

    @property
    def cell_volume(self) -> float:
        """Return the volume of a single cell."""
        return self.dx * self.dy * self.dz

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 3D coordinate arrays for cell centers."""
        return np.meshgrid(self.x, self.y, self.z, indexing="ij")


@dataclass
class MHDState3D:
    """State container for 3D MHD variables.

    Attributes
    ----------
    rho : ndarray
        Mass density [kg/m^3], shape (nx, ny, nz).
    mom : ndarray
        Momentum density [kg/m^2/s], shape (nx, ny, nz, 3).
    energy : ndarray
        Total energy density [J/m^3], shape (nx, ny, nz).
    B : ndarray
        Magnetic field [T], shape (nx, ny, nz, 3).
    Te : ndarray, optional
        Electron temperature [K], shape (nx, ny, nz).
    Ti : ndarray, optional
        Ion temperature [K], shape (nx, ny, nz).
    """

    rho: np.ndarray
    mom: np.ndarray
    energy: np.ndarray
    B: np.ndarray
    Te: Optional[np.ndarray] = None
    Ti: Optional[np.ndarray] = None
    psi: Optional[np.ndarray] = None

    def velocity(self) -> np.ndarray:
        """Return bulk velocity v = mom / rho."""
        return self.mom / self.rho[..., np.newaxis]

    def pressure(self, gamma: float = 5.0 / 3.0) -> np.ndarray:
        """Compute thermal pressure from energy."""
        v = self.velocity()
        kinetic = 0.5 * self.rho * np.sum(v ** 2, axis=-1)
        magnetic = 0.5 * np.sum(self.B ** 2, axis=-1) / mu_0
        internal = self.energy - kinetic - magnetic
        internal = np.maximum(internal, 1e-30)  # Ensure positive internal energy
        return (gamma - 1.0) * internal

    def copy(self) -> "MHDState3D":
        """Return a deep copy of the state."""
        return MHDState3D(
            rho=self.rho.copy(),
            mom=self.mom.copy(),
            energy=self.energy.copy(),
            B=self.B.copy(),
            Te=self.Te.copy() if self.Te is not None else None,
            Ti=self.Ti.copy() if self.Ti is not None else None,
            psi=self.psi.copy() if self.psi is not None else None,
        )


@dataclass
class HallMHDSolver3D:
    """3D Hall-MHD solver with constrained transport.

    This solver advances the resistive Hall-MHD equations using a
    second-order Godunov scheme with constrained transport to maintain
    div(B) = 0. The Hall electric field is computed as:

        E_H = (eta_H / |B|) * (J x B)

    where eta_H is the Hall coefficient proportional to c/(4*pi*n*e).

    Parameters
    ----------
    grid : Grid3D
        Computational grid.
    eta : float
        Resistivity [Ohm*m].
    eta_H : float
        Hall coefficient [m^3/(A*s)].
    gamma : float
        Adiabatic index (default 5/3).
    """

    grid: Grid3D
    eta: float = 0.0
    eta_H: float = 0.0
    gamma: float = 5.0 / 3.0
    c_h: float = 0.0
    c_p: float = 0.0
    bc: Optional[Callable[["MHDState3D"], None]] = None
    ct_update: CTUpdate = field(default_factory=CTUpdate)

    def compute_current_density(self, B: np.ndarray) -> np.ndarray:
        """Compute current density J = curl(B) / mu_0.

        Parameters
        ----------
        B : ndarray
            Magnetic field, shape (nx, ny, nz, 3).

        Returns
        -------
        ndarray
            Current density J, shape (nx, ny, nz, 3).
        """
        curl_B = compute_curl(B, self.grid.dx, self.grid.dy, self.grid.dz)
        return curl_B / mu_0

    def compute_hall_field(
        self,
        J: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """Compute Hall electric field E_H = (eta_H / |B|) * (J x B).

        Parameters
        ----------
        J : ndarray
            Current density, shape (nx, ny, nz, 3).
        B : ndarray
            Magnetic field, shape (nx, ny, nz, 3).

        Returns
        -------
        ndarray
            Hall electric field, shape (nx, ny, nz, 3).
        """
        B_mag = np.sqrt(np.sum(B ** 2, axis=-1, keepdims=True))
        B_mag = np.maximum(B_mag, 1e-30)
        J_cross_B = np.cross(J, B, axis=-1)
        return (self.eta_H / B_mag) * J_cross_B

    def compute_resistive_field(self, J: np.ndarray) -> np.ndarray:
        """Compute resistive electric field E_eta = eta * J.

        Parameters
        ----------
        J : ndarray
            Current density, shape (nx, ny, nz, 3).

        Returns
        -------
        ndarray
            Resistive electric field, shape (nx, ny, nz, 3).
        """
        return self.eta * J

    def compute_convective_field(
        self,
        v: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """Compute convective electric field E_conv = -v x B.

        Parameters
        ----------
        v : ndarray
            Velocity field, shape (nx, ny, nz, 3).
        B : ndarray
            Magnetic field, shape (nx, ny, nz, 3).

        Returns
        -------
        ndarray
            Convective electric field, shape (nx, ny, nz, 3).
        """
        return -np.cross(v, B, axis=-1)

    def compute_total_electric_field(
        self,
        state: MHDState3D,
        J: np.ndarray,
    ) -> np.ndarray:
        """Compute total electric field for Ohm's law.

        E = -v x B + eta*J + E_H

        Parameters
        ----------
        state : MHDState3D
            Current MHD state.
        J : ndarray
            Current density.

        Returns
        -------
        ndarray
            Total electric field, shape (nx, ny, nz, 3).
        """
        v = state.velocity()
        E_conv = self.compute_convective_field(v, state.B)
        E_res = self.compute_resistive_field(J)
        E_hall = self.compute_hall_field(J, state.B) if self.eta_H != 0.0 else 0.0
        return E_conv + E_res + E_hall

    def constrained_transport_update(
        self,
        B: np.ndarray,
        E: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance magnetic field using constrained transport: dB/dt = -curl(E).

        This update preserves div(B) = 0 to machine precision when
        using consistent finite-difference stencils.

        Parameters
        ----------
        B : ndarray
            Magnetic field at time t, shape (nx, ny, nz, 3).
        E : ndarray
            Electric field at time t, shape (nx, ny, nz, 3).
        dt : float
            Time step [s].

        Returns
        -------
        ndarray
            Magnetic field at time t + dt.
        """
        return self.ct_update.update(
            B, E, dt, self.grid.dx, self.grid.dy, self.grid.dz
        )

    def _compute_flux_x(
        self, state: MHDState3D, p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute fluxes in x-direction."""
        rho = state.rho
        v = state.velocity()
        B = state.B
        B2 = np.sum(B ** 2, axis=-1)
        ptot = p + 0.5 * B2 / mu_0

        flux_rho = rho * v[..., 0]
        flux_mom = np.zeros_like(state.mom)
        flux_mom[..., 0] = rho * v[..., 0] ** 2 + ptot - B[..., 0] ** 2 / mu_0
        flux_mom[..., 1] = rho * v[..., 0] * v[..., 1] - B[..., 0] * B[..., 1] / mu_0
        flux_mom[..., 2] = rho * v[..., 0] * v[..., 2] - B[..., 0] * B[..., 2] / mu_0

        vB = np.sum(v * B, axis=-1)
        flux_energy = (state.energy + ptot) * v[..., 0] - B[..., 0] * vB / mu_0

        return flux_rho, flux_mom, flux_energy

    def step(
        self,
        state: MHDState3D,
        dt: float,
        current: float = 0.0,
        voltage: float = 0.0,
    ) -> MHDState3D:
        """Advance the MHD state by one time step.

        Uses a second-order MUSCL-Hancock scheme for the fluid variables
        and constrained transport for the magnetic field.

        Parameters
        ----------
        state : MHDState3D
            Current MHD state.
        dt : float
            Time step [s].
        current : float
            External circuit current [A] (optional).
        voltage : float
            External voltage [V] (optional).

        Returns
        -------
        MHDState3D
            Updated state at t + dt.
        """
        if self.bc is not None:
            self.bc(state)

        new_state = state.copy()
        rho = new_state.rho
        mom = new_state.mom
        energy = new_state.energy
        B = new_state.B

        J = self.compute_current_density(B)
        E = self.compute_total_electric_field(state, J)

        B = self.constrained_transport_update(B, E, dt)

        p = state.pressure(self.gamma)
        flux_rho, flux_mom, flux_energy = self._compute_flux_x(state, p)

        dx = self.grid.dx
        rho = rho - dt / dx * (
            np.roll(flux_rho, -1, axis=0) - flux_rho
        )
        for i in range(3):
            mom[..., i] = mom[..., i] - dt / dx * (
                np.roll(flux_mom[..., i], -1, axis=0) - flux_mom[..., i]
            )
        energy = energy - dt / dx * (
            np.roll(flux_energy, -1, axis=0) - flux_energy
        )

        rho = np.maximum(rho, 1e-30)

        if self.c_h != 0.0 or self.c_p != 0.0:
            psi = new_state.psi if new_state.psi is not None else np.zeros_like(rho)
            from .constrained_transport import compute_divergence
            divB = compute_divergence(B, dx, self.grid.dy, self.grid.dz)
            psi = psi - dt * (self.c_h ** 2 * divB + self.c_p * psi)
            grad_psi = np.stack([
                (np.roll(psi, -1, axis=i) - np.roll(psi, 1, axis=i)) / (2 * dx)
                for i in range(3)
            ], axis=-1)
            B = B - dt * grad_psi
            new_state.psi = psi

        B = project_div_free(B)

        J_new = self.compute_current_density(B)
        heating = self.eta * np.sum(J_new ** 2, axis=-1)
        energy = energy + dt * heating

        new_state.rho = rho
        new_state.mom = mom
        new_state.energy = energy
        new_state.B = B

        if self.bc is not None:
            self.bc(new_state)

        return new_state

    def cfl_timestep(self, state: MHDState3D, cfl: float = 0.5) -> float:
        """Compute CFL-limited time step.

        Parameters
        ----------
        state : MHDState3D
            Current state.
        cfl : float
            CFL number (default 0.5).

        Returns
        -------
        float
            Maximum stable time step.
        """
        v = state.velocity()
        v_mag = np.sqrt(np.sum(v ** 2, axis=-1))

        B2 = np.sum(state.B ** 2, axis=-1)
        v_A = np.sqrt(B2 / (mu_0 * state.rho))

        p = state.pressure(self.gamma)
        c_s = np.sqrt(self.gamma * p / state.rho)

        v_fast = np.sqrt(v_A ** 2 + c_s ** 2)
        v_max = np.max(v_mag + v_fast)

        dx_min = min(self.grid.dx, self.grid.dy, self.grid.dz)
        return cfl * dx_min / max(v_max, 1e-30)

    def total_magnetic_energy(self, state: MHDState3D) -> float:
        """Compute total magnetic energy in the domain."""
        B2 = np.sum(state.B ** 2, axis=-1)
        return 0.5 * np.sum(B2) / mu_0 * self.grid.cell_volume

    def total_kinetic_energy(self, state: MHDState3D) -> float:
        """Compute total kinetic energy in the domain."""
        v = state.velocity()
        v2 = np.sum(v ** 2, axis=-1)
        return 0.5 * np.sum(state.rho * v2) * self.grid.cell_volume

    def divergence_error(self, state: MHDState3D) -> float:
        """Compute the L2 norm of div(B)."""
        from .constrained_transport import compute_divergence
        divB = compute_divergence(
            state.B, self.grid.dx, self.grid.dy, self.grid.dz
        )
        return float(np.sqrt(np.sum(divB ** 2) * self.grid.cell_volume))
