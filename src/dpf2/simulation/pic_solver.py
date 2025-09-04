# pic_solver.py

"""
Particle-in-Cell (PIC) solver implementing classical electromagnetic dynamics,
basic collision processes, diagnostics and optional coupling to WarpX.

Recent additions include:
- Energy spectra and phase-space diagnostic routines
- Configurable particle boundary conditions
- Hooks for quantum and radiation models
- Pluggable mesh adaptivity
"""

from __future__ import annotations

import numpy as np
import logging
import json
import threading
import math
from numba import njit, prange
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
from ..core.bases import CouplingState
from ..diagnostics import synthetic_signals
from .config_schema import PICConfig
from .models import PhysicsModule
from .utils import FieldManager, SimulationState
from .warpx_wrapper import WarpXWrapper
from .collision_model import (
    CollisionProcess,
    BetheBlochStopping,
    ElectronIonCollision,
    ElectronNeutralCollision,
    IonizationProcess,
    RecombinationProcess,
)

# Configure logger
logger = logging.getLogger('pic_solver')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

#-----------------------------------------------------------------------------------------
# Instability and resistivity models
#-----------------------------------------------------------------------------------------
class MZeroInstability:
    """Simple exponential m=0 instability growth model."""

    def __init__(self, growth_rate: float = 0.0):
        self.growth_rate = growth_rate

    def apply(self, Ez: "np.ndarray", dt: float) -> "np.ndarray":
        if self.growth_rate == 0.0:
            return Ez
        return Ez * np.exp(self.growth_rate * dt)


class AnomalousResistivity:
    """Mechanism-based anomalous resistivity model."""

    def __init__(self, eta: float = 0.0, j_crit: float = 1.0):
        self.eta = eta
        self.j_crit = j_crit

    def apply(self, E: "np.ndarray", J: "np.ndarray") -> "np.ndarray":
        if self.eta == 0.0:
            return E
        magJ = np.linalg.norm(J, axis=0)
        mask = magJ > self.j_crit
        if np.any(mask):
            E[:, mask] -= self.eta * J[:, mask]
        return E

#-----------------------------------------------------------------------------------------
# PIC Solver Class
#-----------------------------------------------------------------------------------------
class PICSolver(PhysicsModule):
    """Classical PIC solver with optional WarpX coupling.

    Limitations:
        - Lacks full quantum or radiation models.
        - Diagnostics are currently minimal.
    """

    # Physical and solver constants
    c = 299_792_458.0            # Speed of light in vacuum (m/s)
    epsilon0 = 8.854187817e-12   # Vacuum permittivity (F/m)
    mu0 = 4 * np.pi * 1e-7       # Vacuum permeability (H/m)
    k_B = 1.380649e-23           # Boltzmann constant (J/K)

    # PML and Maxwell solver defaults
    pml_thickness = 10
    pml_sigma_max = 1.0
    maxwell_order = 4
    default_shape = (1, 1, 1)

    # Miscellaneous constants
    ionization_energy = 13.6     # eV, used for Bethe-Bloch stopping

    def __init__(self, config: PICConfig, field_manager: FieldManager):
        """
        Initializes the PICSolver with configuration parameters.

        Args:
            config: A PICConfig object containing the solver parameters.
            field_manager: A FieldManager object for managing electromagnetic fields.
        """
        self.config = config
        self.nx, self.ny, self.nz = config.grid_shape
        self.dx, self.dy, self.dz = config.grid_spacing
        self.origin = (0.0, 0.0, 0.0)
        self.max_dt = config.max_dt
        self.em = config.electromag
        self.boundary_conditions = config.boundary_conditions
        bc_map = {'periodic': 0, 'reflecting': 1, 'absorbing': 2}
        self.bc_codes = (
            bc_map.get(self.boundary_conditions.get('x', 'reflecting'), 1),
            bc_map.get(self.boundary_conditions.get('y', 'reflecting'), 1),
            bc_map.get(self.boundary_conditions.get('z', 'reflecting'), 1),
        )
        self.bc_periodic = tuple(code == 0 for code in self.bc_codes)
        self.dt = config.dt if config.dt else 0.5 * min(self.dx, self.dy, self.dz) / PICSolver.c
        self.use_warpx = config.use_warpx
        self.unity_params = config.unity_params or {}
        self.vdf_bins = config.vdf_bins
        self.max_vel = config.max_vel
        self.subgrid_resolution = config.subgrid_resolution
        self.amr = config.amr
        self.density_threshold = config.density_threshold
        self.levels = [{'grid_shape': tuple(config.grid_shape), 'grid_spacing': tuple(config.grid_spacing), 'offset': (0, 0, 0)}]
        self.heating_grid = 0.0
        self.species = {}
        self.vdf = {}
        self.field_manager = field_manager
        self.collisions: List[CollisionProcess] = []
        self.collisions.extend([
            BetheBlochStopping('ion', Z_eff=1, I_mean_ev=PICSolver.ionization_energy),
            ElectronIonCollision(), ElectronNeutralCollision(),
            IonizationProcess(), RecombinationProcess()
        ])
        self.enable_quantum = config.enable_quantum
        self.enable_radiation = config.enable_radiation
        self.quantum_model: Optional[Callable[["PICSolver"], None]] = None
        self.radiation_model: Optional[Callable[["PICSolver"], None]] = None
        self.mesh_adapter: Optional[Callable[[], None]] = self.refine_grid if config.enable_mesh_adaptivity else None
        self.warpx = WarpXWrapper(
            config.grid_shape, config.grid_spacing, config.electromag,
            PICSolver.pml_thickness, PICSolver.pml_sigma_max,
            PICSolver.maxwell_order, PICSolver.default_shape, self,
            enable_amr=config.amr
        ) if config.use_warpx else None
        self.coupling_state = CouplingState()
        self.history: List[CouplingState] = []
        self.m0_instability_model: Optional[MZeroInstability] = None
        self.anomalous_resistivity_model: Optional[AnomalousResistivity] = None
        logger.info('PIC solver initialized')

    def add_species(self, name, charge, mass, positions, velocities):
        """Adds a new particle species to the simulation."""
        try:
            if name in self.species: raise ValueError(f"Species {name} exists")
            pos, vel = np.array(positions, float), np.array(velocities, float)
            assert pos.shape == vel.shape and pos.shape[1] == 3
            self.species[name] = {'q': charge, 'm': mass, 'pos': pos, 'vel': vel} # Store species data
            self.vdf[name] = np.zeros((self.vdf_bins,) * 3)
            logger.info(f"Added species {name} N={pos.shape[0]}")
        except Exception as e:
            logger.error(f"Error adding species {name}: {e}")

    def set_m0_instability(self, growth_rate: float):
        """Enable a simple m=0 instability growth model."""
        self.m0_instability_model = MZeroInstability(growth_rate)

    def set_anomalous_resistivity(self, eta: float, j_crit: float = 1.0):
        """Enable mechanism-based anomalous resistivity."""
        self.anomalous_resistivity_model = AnomalousResistivity(eta, j_crit)

    #-------------------------------------------------------------------------------------
    # Boris pusher
    #-------------------------------------------------------------------------------------
    @njit(parallel=True, fastmath=True)
    def boris_push_numba(self, pos, vel, q, m, dt, E, B, origin, dxyz, bc, dims):
        """
        Numba-accelerated Boris pusher for updating particle positions and velocities.

        Args:
            pos: Particle positions.
            vel: Particle velocities.
            q: Particle charge.
            m: Particle mass.
            dt: Time step.
            E: Electric field.
            B: Magnetic field.
            origin: Origin of the grid.
            dxyz: Grid spacing (dx, dy, dz).
            bc: Tuple describing boundary conditions per axis (0=periodic,
                1=reflecting, 2=absorbing).
            dims: Grid dimensions (nx, ny, nz).
        """
        nx,ny,nz=dims; dx,dy,dz=dxyz; ox,oy,oz=origin
        for idx in prange(pos.shape[0]):
            x,y,z=pos[idx]; xi=(x-ox)/dx; yi=(y-oy)/dy; zi=(z-oz)/dz
            i0,j0,k0=int(np.floor(xi)),int(np.floor(yi)),int(np.floor(zi))
            fx,fy,fz=xi-i0,yi-j0,zi-k0
            Ex=Ey=Ez=Bx=By=Bz=0.0
            for di in (0,1):
                wx=(1-fx) if di==0 else fx; ii=(i0+di)%nx if bc[0]==0 else i0+di
                if ii<0 or ii>=nx: continue
                for dj in (0,1):
                    wy=(1-fy) if dj==0 else fy; jj=(j0+dj)%ny if bc[1]==0 else j0+dj
                    if jj<0 or jj>=ny: continue
                    for dk in (0,1):
                        wz=(1-fz) if dk==0 else fz; kk=(k0+dk)%nz if bc[2]==0 else k0+dk
                        if kk<0 or kk>=nz: continue
                        w=wx*wy*wz
                        Ex+=w*E[0,ii,jj,kk]; Ey+=w*E[1,ii,jj,kk]; Ez+=w*E[2,ii,jj,kk]
                        Bx+=w*B[0,ii,jj,kk]; By+=w*B[1,ii,jj,kk]; Bz+=w*B[2,ii,jj,kk]
            vx,vy,vz=vel[idx]
            gamma=math.sqrt(1+(vx*vx+vy*vy+vz*vz)/PICSolver.c**2)
            ux,uy,uz=gamma*vx,gamma*vy,gamma*vz
            ux+=q/m*Ex*(dt*0.5); uy+=q/m*Ey*(dt*0.5); uz+=q/m*Ez*(dt*0.5)
            gamma_star=math.sqrt(1+(ux*ux+uy*uy+uz*uz)/PICSolver.c**2)
            coeff=(q/m)*(dt*0.5)/gamma_star
            tx,ty,tz=coeff*Bx,coeff*By,coeff*Bz
            upr_x=ux+(uy*tz-uz*ty); upr_y=uy+(uz*tx-ux*tz); upr_z=uz+(ux*ty-uy*tx)
            s_den=1+tx*tx+ty*ty+tz*tz; sx,sy,sz=2*tx/s_den,2*ty/s_den,2*tz/s_den
            uxp=ux+(upr_y*sz-upr_z*sy); uyp=uy+(upr_z*sx-upr_x*sz); uzp=uz+(upr_x*sy-upr_y*sx)
            ux_new=uxp+q/m*Ex*(dt*0.5); uy_new=uyp+q/m*Ey*(dt*0.5); uz_new=uzp+q/m*Ez*(dt*0.5)
            g_new=math.sqrt(1+(ux_new*ux_new+uy_new*uy_new+uz_new*uz_new)/PICSolver.c**2)
            vel[idx,0]=ux_new/g_new; vel[idx,1]=uy_new/g_new; vel[idx,2]=uz_new/g_new
            pos[idx,0]+=vel[idx,0]*dt; pos[idx,1]+=vel[idx,1]*dt; pos[idx,2]+=vel[idx,2]*dt
            for d,(low,high,length) in enumerate(((ox,ox+nx*dx,nx*dx),(oy,oy+ny*dy,ny*dy),(oz,oz+nz*dz,nz*dz))):
                b=bc[d]
                if b==0:  # periodic
                    if pos[idx,d]<low: pos[idx,d]+=length
                    if pos[idx,d]>=high: pos[idx,d]-=length
                elif b==1:  # reflecting
                    if pos[idx,d]<low:
                        pos[idx,d]=2*low-pos[idx,d]; vel[idx,d]=-vel[idx,d]
                    if pos[idx,d]>high:
                        pos[idx,d]=2*high-pos[idx,d]; vel[idx,d]=-vel[idx,d]
                elif b==2:  # absorbing
                    if pos[idx,d]<low or pos[idx,d]>high:
                        pos[idx,d]=np.nan; vel[idx,d]=0.0

    #-------------------------------------------------------------------------------------
    # Charge & Current deposition
    #-------------------------------------------------------------------------------------
    def deposit_charge(self):
        """Deposits particle charge onto the grid."""
        try:
            self.field_manager.rho.fill(0)
            vol = self.dx * self.dy * self.dz
            bcx, bcy, bcz = self.bc_periodic
            for spc in self.species.values():  # Loop over species
                q, pos = spc['q'], spc['pos']
                for p in pos:
                    xi, yi, zi = (p[0] - self.origin[0]) / self.dx, (p[1] - self.origin[1]) / self.dy, (p[2] - self.origin[2]) / self.dz
                    i0, j0, k0 = int(np.floor(xi)), int(np.floor(yi)), int(np.floor(zi))
                    fx, fy, fz = xi - i0, yi - j0, zi - k0
                    for di in (0, 1):
                        wx = (1 - fx) if di == 0 else fx; ii = (i0 + di) % self.nx if bcx else i0 + di
                        if ii < 0 or ii >= self.nx:
                            continue
                        for dj in (0, 1):
                            wy = (1 - fy) if dj == 0 else fy; jj = (j0 + dj) % self.ny if bcy else j0 + dj
                            if jj < 0 or jj >= self.ny:
                                continue
                            for dk in (0, 1):
                                wz = (1 - fz) if dk == 0 else fz; kk = (k0 + dk) % self.nz if bcz else k0 + dk
                                if kk < 0 or kk >= self.nz:
                                    continue
                                self.field_manager.rho[ii, jj, kk] += q * (wx * wy * wz) / vol
        except Exception as e:
            logger.error(f"Error depositing charge: {e}")

    def deposit_current(self):
        """Deposits particle current onto the grid."""
        try:
            self.field_manager.J.fill(0)
            vol = self.dx * self.dy * self.dz
            bcx, bcy, bcz = self.bc_periodic
            for spc in self.species.values():  # Loop over species
                q, pos, vel = spc['q'], spc['pos'], spc['vel']
                for p, v in zip(pos, vel):
                    xi, yi, zi = (p[0] - self.origin[0]) / self.dx, (p[1] - self.origin[1]) / self.dy, (p[2] - self.origin[2]) / self.dz
                    i0, j0, k0 = int(np.floor(xi)), int(np.floor(yi)), int(np.floor(zi))
                    fx, fy, fz = xi - i0, yi - j0, zi - k0
                    for di in (0, 1):
                        wx = (1 - fx) if di == 0 else fx; ii = (i0 + di) % self.nx if bcx else i0 + di
                        if ii < 0 or ii >= self.nx:
                            continue
                        for dj in (0, 1):
                            wy = (1 - fy) if dj == 0 else fy; jj = (j0 + dj) % self.ny if bcy else j0 + dj
                            if jj < 0 or jj >= self.ny:
                                continue
                            for dk in (0, 1):
                                wz = (1 - fz) if dk == 0 else fz; kk = (k0 + dk) % self.nz if bcz else k0 + dk
                                if kk < 0 or kk >= self.nz:
                                    continue
                                w = wx * wy * wz
                                self.field_manager.J[0, ii, jj, kk] += q * v[0] * w / vol
                                self.field_manager.J[1, ii, jj, kk] += q * v[1] * w / vol
                                self.field_manager.J[2, ii, jj, kk] += q * v[2] * w / vol
        except Exception as e:
            logger.error(f"Error depositing current: {e}")

    def filter_current(self):
        """Applies a digital filter to the current density."""
        try:
            kernel = np.array([0.25, 0.5, 0.25])
            J = self.field_manager.get_J()
            for c in range(3):
                for ax in range(3):
                    J[c] = np.apply_along_axis(lambda arr: np.convolve(arr, kernel, 'same'), ax, J[c]) # Apply filter along each axis
            self.field_manager.update_J(J)
        except Exception as e:
            logger.error(f"Error filtering current: {e}")

    #-------------------------------------------------------------------------------------
    # PML, field update, divergence cleaning
    #-------------------------------------------------------------------------------------
    def _init_pml(self):
        """Initializes the Perfectly Matched Layer (PML) parameters."""
        t = PICSolver.pml_thickness
        prof = PICSolver.pml_sigma_max * (np.linspace(0, 1, t)**2)
        self.pml_sigma_e = np.zeros(self.nz)
        self.pml_sigma_e[:t] = prof[::-1]; self.pml_sigma_e[-t:] = prof
        self.pml_sigma_b = self.pml_sigma_e.copy()

    def _apply_pml(self):
        """Applies PML damping to the electromagnetic fields."""
        try:
            E = self.field_manager.get_E()
            B = self.field_manager.get_B()
            σe = self.pml_sigma_e[np.newaxis, np.newaxis, :]
            σb = self.pml_sigma_b[np.newaxis, np.newaxis, :]
            E *= np.exp(-σe * self.dt / PICSolver.epsilon0)  # Apply PML damping
            B *= np.exp(-σb * self.dt / PICSolver.mu0)
            self.field_manager.update_E(E)
            self.field_manager.update_B(B)
        except Exception as e:
            logger.error(f"Error applying PML: {e}")

    def _clean_divergence(self):
        """Cleans the divergence of the electric field using a Poisson solver."""
        try:
            rho = self.field_manager.get_rho()
            E = self.field_manager.get_E()
            rho_hat = np.fft.fftn(rho / PICSolver.epsilon0)
            divE = (np.gradient(E[0], self.dx, axis=0) +
                    np.gradient(E[1], self.dy, axis=1) +
                    np.gradient(E[2], self.dz, axis=2))
            divE_hat = np.fft.fftn(divE)
            kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
            ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
            kz = 2 * np.pi * np.fft.fftfreq(self.nz, self.dz)
            k2 = np.add.outer(np.add.outer(kx**2, ky**2), kz**2)
            k2[0, 0, 0] = 1
            phi_hat = (divE_hat - rho_hat) / (-k2)  # Solve Poisson equation in Fourier space
            for i, arr in enumerate((kx, ky, kz)):
                grad_phi = np.fft.ifftn(1j * arr.reshape([arr.size if j == i else 1 for j in range(3)]) * phi_hat).real
                E[i] -= grad_phi
            self.field_manager.update_E(E)
        except Exception as e:
            logger.error(f"Error cleaning divergence: {e}")

    def solve_fields(self):
        """Updates the electromagnetic fields using FDTD or WarpX."""
        try:
            E = self.field_manager.get_E()
            B = self.field_manager.get_B()
            rho = self.field_manager.get_rho()
            J = self.field_manager.get_J()
            if self.warpx:
                E, B = self.warpx.step(rho, J, E, B, self.dt)
                if self.anomalous_resistivity_model:
                    E = self.anomalous_resistivity_model.apply(E, J)
                if self.m0_instability_model:
                    E[2] = self.m0_instability_model.apply(E[2], self.dt)
            else:
                # FDTD update
                curlE = np.array([(np.roll(E[2], -1, 1) - E[2]) / self.dy - (np.roll(E[1], -1, 2) - E[1]) / self.dz,
                                  (np.roll(E[0], -1, 2) - E[0]) / self.dz - (np.roll(E[2], -1, 0) - E[2]) / self.dx,
                                  (np.roll(E[1], -1, 0) - E[1]) / self.dx - (np.roll(E[0], -1, 1) - E[0]) / self.dy])
                B -= self.dt * curlE
                curlB = np.array([(np.roll(B[2], -1, 1) - B[2]) / self.dy - (np.roll(B[1], -1, 2) - B[1]) / self.dz,
                                  (np.roll(B[0], -1, 2) - B[0]) / self.dz - (np.roll(B[2], -1, 0) - B[2]) / self.dx,
                                  (np.roll(B[1], -1, 0) - B[1]) / self.dx - (np.roll(E[0], -1, 1) - E[0]) / self.dy])
                E += self.dt * (PICSolver.c**2 * curlB - J / PICSolver.epsilon0)
                if self.anomalous_resistivity_model:
                    E = self.anomalous_resistivity_model.apply(E, J)
                if self.m0_instability_model:
                    E[2] = self.m0_instability_model.apply(E[2], self.dt)
                self._apply_pml()  # Apply PML damping
            self.field_manager.update_E(E)
            self.field_manager.update_B(B)
            self.filter_current()
            self._clean_divergence()
        except Exception as e:
            logger.error(f"Error solving fields: {e}")

    #-------------------------------------------------------------------------------------
    # Diagnostics: VDF, moments, spatial diagnostics
    #-------------------------------------------------------------------------------------
    def calculate_vdf(self):
        """Calculates the velocity distribution function (VDF) for each species."""
        try:
            for name, spc in self.species.items():
                vdf = np.zeros((self.vdf_bins,) * 3)
                dv = 2 * self.max_vel / self.vdf_bins  # Velocity bin size
                for v in spc['vel']:
                    ix = int((v[0] + self.max_vel) // dv)
                    iy = int((v[1] + self.max_vel) // dv)
                    iz = int((v[2] + self.max_vel) // dv)
                    if 0 <= ix < self.vdf_bins and 0 <= iy < self.vdf_bins and 0 <= iz < self.vdf_bins:
                        vdf[ix, iy, iz] += 1
                self.vdf[name] = vdf / (len(spc['vel']) * dv**3)
        except Exception as e:
            logger.error(f"Error calculating VDF: {e}")

    def calculate_moments(self):
        """Calculates velocity moments (average velocity, temperature) for each species."""
        try:
            for name, spc in self.species.items():
                vel = spc['vel']
                avg_vel = np.mean(vel, axis=0)  # Average velocity
                temp = np.mean(np.sum((vel - avg_vel)**2, axis=1)) * spc['m'] / (3 * PICSolver.k_B)
                logger.info(f"Species {name}: <v>={avg_vel}, T={temp:.3e} K")
        except Exception as e:
            logger.error(f"Error calculating moments: {e}")

    def calculate_spatial_diagnostics(self):
        """Calculates spatial diagnostics (density distribution) for each species."""
        try:
            for name, spc in self.species.items():
                pos = spc['pos']
                subgrid_shape = (self.nx // self.subgrid_resolution[0],
                                 self.ny // self.subgrid_resolution[1],
                                 self.nz // self.subgrid_resolution[2])
                density = np.zeros(subgrid_shape)
                for p in pos:  # Loop over particles
                    idx = (int((p[0] - self.origin[0]) // (self.dx * self.subgrid_resolution[0])),
                           int((p[1] - self.origin[1]) // (self.dy * self.subgrid_resolution[1])),
                           int((p[2] - self.origin[2]) // (self.dz * self.subgrid_resolution[2])))
                    if all(0 <= i < s for i, s in zip(idx, subgrid_shape)):
                        density[idx] += 1
                logger.info(f"Species {name}: spatial diagnostics calculated.")
        except Exception as e:
            logger.error(f"Error calculating spatial diagnostics: {e}")

    #-------------------------------------------------------------------------------------
    # AMR
    #-------------------------------------------------------------------------------------
    def refine_grid(self):
        """Refines the grid based on particle density."""
        try:
            if not self.amr:
                return
            rho = self.field_manager.get_rho()
            new_levels = []
            for level in self.levels:
                grid_shape = level['grid_shape']
                grid_spacing = level['grid_spacing']
                offset = level['offset']
                if any(s < 4 for s in grid_shape):
                    continue  # Minimum grid size
                new_grid_shape = tuple(s // 2 for s in grid_shape)
                new_grid_spacing = tuple(s * 2 for s in grid_spacing)
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            new_offset = (offset[0] + i * new_grid_shape[0],
                                          offset[1] + j * new_grid_shape[1],
                                          offset[2] + k * new_grid_shape[2])  # New offset
                            if np.mean(rho[new_offset[0]:new_offset[0]+new_grid_shape[0],
                                                new_offset[1]:new_offset[1]+new_grid_shape[1],
                                                new_offset[2]:new_offset[2]+new_grid_shape[2]]) > self.density_threshold:
                                new_levels.append({'grid_shape': new_grid_shape,
                                                   'grid_spacing': new_grid_spacing,
                                                   'offset': new_offset})
            self.levels.extend(new_levels)
            logger.info(f"AMR: grid refined to {len(self.levels)} levels.")
        except Exception as e:
            logger.error(f"Error refining grid: {e}")

    #-------------------------------------------------------------------------------------
    # Main step
    #-------------------------------------------------------------------------------------
    def step(self, current: float = 0.0, voltage: float = 0.0):
        """Advances the PIC simulation by one time step."""
        try:
            self.deposit_charge()
            self.deposit_current()
            self.solve_fields()
            E = self.field_manager.get_E()
            if voltage:
                E[2] += voltage / (self.nz * self.dz)
                self.field_manager.update_E(E)
            B = self.field_manager.get_B()
            for name, spc in self.species.items():  # Loop over species
                self.boris_push_numba(spc['pos'], spc['vel'], spc['q'], spc['m'], self.dt,
                                      E, B, self.origin, (self.dx, self.dy, self.dz),
                                      self.bc_codes, (self.nx, self.ny, self.nz))
            self.apply_collisions()
            if self.enable_quantum and self.quantum_model:
                self.quantum_model()
            if self.enable_radiation and self.radiation_model:
                self.radiation_model()
            self.calculate_vdf()
            self.calculate_moments()
            self.calculate_spatial_diagnostics()
            if self.mesh_adapter:
                self.mesh_adapter()
            self.stream_to_unity()
            axial_E = float(np.mean(E[2]))
            length = self.nz * self.dz
            emf = axial_E * length
            self.coupling_state = CouplingState(emf=emf, current=current, voltage=voltage, back_reaction=emf)
            self.history.append(self.coupling_state)
        except Exception as e:
            logger.error(f"Error during PIC step: {e}")

    def apply(self, state: SimulationState, dt: float) -> None:  # type: ignore[override]
        """Interface required by ``PhysicsModule``."""
        self.step()

    def apply_collisions(self):
        """Applies collision processes to the particles."""
        try:
            for collision in self.collisions:  # Loop over collision processes
                collision.apply(self, self.dt)
        except Exception as e:
            logger.error(f"Error applying collisions: {e}")

    def compute_optimal_dt(self):
        """Computes an optimal time step based on CFL and plasma frequency."""
        try:
            rho = self.field_manager.get_rho()
            max_v = 0.0
            for spc in self.species.values():  # Loop over species
                max_v = max(max_v, np.max(np.linalg.norm(spc['vel'], axis=1)))
            cfl_dt = 0.5 * min(self.dx, self.dy, self.dz) / max_v if max_v > 0 else float('inf')
            plasma_dt = 0.0
            for spc in self.species.values():
                if spc['q'] != 0:
                    plasma_dt = min(plasma_dt, np.sqrt(PICSolver.epsilon0 * spc['m'] / (spc['q']**2 * np.max(rho))))  # Plasma frequency
            new_dt = min(cfl_dt, plasma_dt)
            if self.max_dt is not None:
                new_dt = min(new_dt, self.max_dt)
            logger.info(f"Optimal dt: {new_dt:.3e} s")
            return new_dt
        except Exception as e:
            logger.error(f"Error computing optimal dt: {e}")
            return self.dt

    def compute_total_energy(self):
        """Computes the total energy in the system."""
        try:
            E = self.field_manager.get_E()
            B = self.field_manager.get_B()
            ke = 0.0
            for spc in self.species.values():  # Loop over species
                ke += 0.5 * spc['m'] * np.sum(np.linalg.norm(spc['vel'], axis=1)**2)
            fe = 0.5 * PICSolver.epsilon0 * np.sum(E**2) * self.dx * self.dy * self.dz
            fm = 0.5 / PICSolver.mu0 * np.sum(B**2) * self.dx * self.dy * self.dz
            logger.info(f"Total energy: KE={ke:.3e} J, FE={fe:.3e} J, FM={fm:.3e} J")
            return ke + fe + fm
        except Exception as e:
            logger.error(f"Error computing total energy: {e}")
            return 0.0

    def coupling_interface(self) -> CouplingState:
        """Return coupling information for circuit solvers."""
        return self.coupling_state

    #-------------------------------------------------------------------------------------
    # Diagnostics
    #-------------------------------------------------------------------------------------
    def compute_energy_spectra(self, species: Optional[str] = None, bins: int = 50):
        """Compute kinetic-energy spectra for particles.

        Args:
            species: Name of the species to analyse. If ``None`` a dictionary of
                spectra for all species is returned.
            bins: Number of histogram bins.

        Returns:
            Tuple of ``(bin_edges, counts)`` for the requested species or a
            dictionary of such tuples when ``species`` is ``None``.
        """
        results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        names = [species] if species else list(self.species.keys())
        for name in names:
            spc = self.species[name]
            m = spc['m']
            vel = spc['vel']
            energy = 0.5 * m * np.sum(vel**2, axis=1)
            hist, edges = np.histogram(energy, bins=bins)
            results[name] = (edges, hist)
        return results[species] if species else results

    def compute_phase_space(self, species: str, x_dim: int = 0, v_dim: int = 0, bins: int = 50):
        """Compute a phase-space histogram for a given species."""
        spc = self.species[species]
        x = spc['pos'][:, x_dim]
        v = spc['vel'][:, v_dim]
        H, xedges, vedges = np.histogram2d(x, v, bins=bins)
        return xedges, vedges, H

    def set_quantum_model(self, model: Callable[["PICSolver"], None]):
        """Attach a quantum-effects model callable."""
        self.quantum_model = lambda: model(self)

    def set_radiation_model(self, model: Callable[["PICSolver"], None]):
        """Attach a radiation model callable."""
        self.radiation_model = lambda: model(self)

    def set_mesh_adapter(self, adapter: Callable[["PICSolver"], None]):
        """Attach a mesh-adaptivity callback."""
        self.mesh_adapter = lambda: adapter(self)

    #-------------------------------------------------------------------------------------
    # Unity streaming
    #-------------------------------------------------------------------------------------
    def stream_to_unity(self):
        """Streams data to Unity for real-time visualization."""
        try:
            if self.unity_ws is None:
                return
            data = {'time': 0.0, 'current': 0.0, 'slice': [], 'particles': []}
            # ... (implementation for streaming data) ...
            self.unity_ws.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error streaming to Unity: {e}")

    def _unity_heartbeat(self):
        """Sends a heartbeat to the Unity WebSocket."""
        try:
            while True:
                import time
                self.unity_ws.send(json.dumps({'heartbeat': True}))
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error in Unity heartbeat: {e}")

    #-------------------------------------------------------------------------------------
    # Validation utilities
    #-------------------------------------------------------------------------------------
    def _load_reference_current(self, device: str, data_dir: Optional[Path] = None) -> np.ndarray:
        base = Path(data_dir) if data_dir else Path('data/validation') / device
        data = np.loadtxt(base / 'current.csv', delimiter=',', skiprows=1)
        return data[:, 1]

    def validate_spike(self, device: str, data_dir: Optional[Path] = None) -> float:
        """Return RMSE between synthetic current and reference spike data."""
        ref = self._load_reference_current(device, data_dir)
        sim = synthetic_signals.current_waveform(self.history)
        n = min(len(ref), len(sim))
        if n == 0:
            return float('inf')
        return float(np.sqrt(np.mean((np.array(sim[:n]) - ref[:n]) ** 2)))

    def validate_pf1000_and_mjolnir(self) -> Dict[str, float]:
        """Validate against PF-1000 and MJOLNIR spike data using synthetic diagnostics."""
        results: Dict[str, float] = {}
        for dev in ('PF1000', 'MJOLNIR'):
            try:
                results[dev] = self.validate_spike(dev)
            except Exception:
                results[dev] = float('inf')
        return results

    #-------------------------------------------------------------------------------------
    # UQ & V&V
    #-------------------------------------------------------------------------------------
    def run_uq_simulation(self, n_samples=100):
        """Runs an Uncertainty Quantification (UQ) simulation."""
        try:
            results = []
            for _ in range(n_samples):
                # Sample parameters from distributions
                # ... (implementation for sampling parameters) ...
                # Create and run a PICSolver instance
                solver = PICSolver(self.config)
                solver.step()
                results.append(solver.compute_total_energy())
            return results
        except Exception as e:
            logger.error(f"Error during UQ simulation: {e}")
            return []

    def run_convergence_study(self, resolutions):
        """Runs a convergence study."""
        try:
            results = []
            for res in resolutions:
                self.config.grid_shape = res
                solver = PICSolver(self.config)
                solver.step()
                results.append(solver.compute_total_energy())
            return results
        except Exception as e:
            logger.error(f"Error during convergence study: {e}")
            return []

    def write_checkpoint(self, filename):
        """Writes a checkpoint of the simulation state."""
        try:
            if self.warpx:
                self.warpx.write_checkpoint(filename)
        except Exception as e:
            logger.error(f"Error writing checkpoint: {e}")

    def restart(self, h5file):
        """Restarts the simulation from a checkpoint."""
        try:
            # ADIOS2 no-op; use HDF5 for particles
            self.warpx.read_checkpoint(h5file)
            with h5py.File(h5file, 'r') as f:
                grp = f['particles']
                for name in self.species:
                    pos = grp[f"{name}_pos"][:]
                    vel = grp[f"{name}_vel"][:]
                    self.warp.clear_particles(name)
                    self.warp.add_particles(pos.tolist(), vel.tolist(), name)
        except Exception as e:
            logger.error(f"Error during restart: {e}")

    def initialize(self):
        """
        Initializes the PIC model.
        """
        logger.info("PICModel initialized.")

    def finalize(self):
        """
        Finalizes the PIC model.
        """
        logger.info("PICModel finalized.")
