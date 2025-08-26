"""Collision model utilities for particle and fluid simulations.

Features
--------
* Coulomb logarithm with quantum diffraction corrections
* Spitzer collision frequencies including electron–neutral collisions
* Implicit electron–ion temperature relaxation
* Energy-dependent cross-section lookup tables
* Braginskii transport coefficient helper
* Simplified D–D fusion rate estimates
* Checkpoint and restart helpers

Future Work
-----------
* Full Fokker–Planck operators and anisotropy relaxation
* Collisional–radiative networks and additional reaction channels
* GPU acceleration and detailed diagnostics
"""

import numpy as np
import h5py
import math
from scipy.interpolate import interp1d, RegularGridInterpolator
from numba import njit, prange, cuda
import logging

try:  # Prefer package-relative imports
    from .models import PhysicsModule, SimulationState  # Import SimulationState
except Exception:  # pragma: no cover - fallback for standalone usage
    from models import PhysicsModule, SimulationState  # type: ignore

import types
from models import PhysicsModule, SimulationState  # Import SimulationState

from typing import List, Dict, Tuple, Optional

logger = logging.getLogger('CollisionModel')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# Physical constants
epsilon0 = 8.854187817e-12  # F/m
kB = 1.380649e-23  # J/K
e_charge = 1.602176634e-19  # C
m_e = 9.10938356e-31  # kg
m_p = 1.67262192369e-27  # kg
pi = math.pi
h_planck = 6.62607015e-34  # J·s
m_d = 3.34358377e-27 # Deuterium mass

# --------------------------------------
# Abstract base for collision operators
# --------------------------------------
class CollisionOperator(PhysicsModule):
    """Interface for collision models.

    Subclasses must provide ``apply`` and ``diagnostics`` implementations.
    This base class performs no conservation checks or algorithmic work."""

    def apply(self, state: SimulationState, dt):
        raise NotImplementedError

    def diagnostics(self, state: SimulationState):
        raise NotImplementedError

    def checkpoint(self):
        raise NotImplementedError

    def restart(self, data):
        raise NotImplementedError

# --------------------------------------
# Coulomb logarithm: strong-weak coupling
# --------------------------------------
@njit
def lnLambda_strong(ne, Te):
    # Debye length
    lambda_D = np.sqrt(epsilon0 * kB * Te / (ne * e_charge**2))
    # classical distance of closest approach
    b_class = e_charge**2 / (4 * pi * epsilon0 * kB * Te)
    # de Broglie wavelength
    lambda_db = h_planck / np.sqrt(2 * pi * m_e * kB * Te)
    b_min = np.maximum(b_class, lambda_db)
    Lambda = lambda_D / (b_min + 1e-30)
    lnL = np.log(np.maximum(Lambda, 1.0))
    if np.any(lnL < 0):
        raise ValueError(f"Unphysical lnLambda<0: min={lnL.min()}")
    return lnL

# --------------------------------------
# Spitzer collision frequencies
# --------------------------------------
@njit
def nu_ei_spitzer(ne, Te, lnL=10.0, Z=1.0):
    coef = 4 * np.sqrt(2 * pi) * ne * Z * e_charge**4 * lnL
    denom = 3 * (4 * pi * epsilon0)**2 * np.sqrt(m_e) * (kB * Te)**1.5
    return coef / (denom + 1e-30)

@njit
def nu_ee(ne, Te, lnL=10.0):
    return nu_ei_spitzer(ne, Te, lnL, Z=1.0) * np.sqrt(2)

@njit
def nu_ii(ni, Ti, lnL=10.0, Z=1.0, mi=m_p):
    coef = 4 * np.sqrt(pi) * ni * Z**4 * e_charge**4 * lnL
    denom = 3 * (4 * pi * epsilon0)**2 * np.sqrt(mi) * (kB * Ti)**1.5
    return coef / (denom + 1e-30)

@njit
def nu_en(ne, Te, nn, sigma_en=1e-19):
    """Electron-neutral collision frequency."""
    v_th_e = np.sqrt(kB * Te / m_e)
    return nn * sigma_en * v_th_e

# --------------------------------------
# Implicit 2x2 electron-ion temperature relaxation
# --------------------------------------
@njit
def relax_ei_implicit(Te, Ti, νei, dt):
    α = νei * dt
    a = 1 + α
    b = -α
    denom = a * a - b * b + 1e-30
    Tsum = Te + Ti
    Te_new = (a * (Te + α * Ti) - b * (Ti + α * Te)) / denom
    Ti_new = (a * (Ti + α * Te) - b * (Te + α * Ti)) / denom
    return Te_new, Ti_new

# --------------------------------------
# Energy-dependent cross-sections (example)
# --------------------------------------
class CrossSectionData:
    """1D tabulated cross-section with simple interpolation.

    The class expects an HDF5 file containing ``energy`` and ``cross_section``
    datasets.  It performs minimal validation and does not track units.
    Missing data results in zero cross-section values."""

    def __init__(self, filename):
        try:
            with h5py.File(filename, 'r') as f:
                self.energy = f['energy'][:]
                self.cross_section = f['cross_section'][:]
            self.interp = interp1d(self.energy, self.cross_section, bounds_error=False, fill_value=0.0)
        except Exception as e:
            logger.error(f"Error loading cross-section data from {filename}: {e}")
            self.energy = []
            self.cross_section = []
            self.interp = lambda E: 0.0  # Default: zero cross-section

    def to_dict(self):
        """Return a serializable representation of the table."""
        return {
            'energy': list(self.energy),
            'cross_section': list(self.cross_section),
        }

    @classmethod
    def from_dict(cls, data):
        """Create a :class:`CrossSectionData` from checkpoint data."""
        obj = cls.__new__(cls)
        obj.energy = np.array(data.get('energy', []))
        obj.cross_section = np.array(data.get('cross_section', []))
        try:
            obj.interp = interp1d(obj.energy, obj.cross_section, bounds_error=False, fill_value=0.0)
        except Exception as e:
            logger.warning(f"Failed to construct cross-section interpolation: {e}")
            obj.interp = lambda E: 0.0
        return obj

    def __call__(self, E):
        return self.interp(E)

# --------------------------------------
# Collision Processes
# --------------------------------------
class CollisionProcess(PhysicsModule):
    """Base class for individual collisional processes.

    Provides only an interface; subclasses handle particle bookkeeping and
    conservation."""

    def apply(self, state: SimulationState, dt):
        raise NotImplementedError

class BetheBlochStopping(CollisionProcess):
    """Simplified Bethe–Bloch stopping power for ions.

    Shell corrections and charge-state evolution are ignored, so results are
    approximate."""

    def __init__(self, name, Z_eff=1, I_mean_ev=13.6, speed_of_light=299792458.0):
        self.name = name
        self.Z_eff = Z_eff
        self.I_mean = I_mean_ev * e_charge  # Convert eV to Joules
        # Allow speed of light to be provided, avoiding dependencies on PICSolver
        self.c = speed_of_light

    def apply(self, state: SimulationState, dt: float):
        """Applies the Bethe-Bloch stopping power to the ions."""
        try:
            if not hasattr(state, "species"):
                return
            for name, spc in state.species.items():
                if name == self.name:
                    vel = spc["vel"]
                    beta = np.linalg.norm(vel, axis=1) / self.c
                    gamma = 1.0 / np.sqrt(1.0 - beta**2)
                    ne = getattr(getattr(state, "field_manager", types.SimpleNamespace(ne=0)), "ne", 0)
                    ne = np.broadcast_to(ne, beta.shape)
                    stopping_power = (
                        4
                        * pi
                        * self.Z_eff**2
                        * e_charge**4
                        * ne
                        / (m_e * self.c**2)
                    ) * (np.log((2 * m_e * self.c**2 * beta**2 * gamma**2) / self.I_mean) - beta**2)
                    vel -= (
                        (stopping_power / (spc["m"] * gamma))[:, np.newaxis]
                        * (vel / (np.linalg.norm(vel, axis=1) + 1e-30)[:, np.newaxis])
                        * dt
                    )
        except Exception as e:
            logger.error(f"Error applying Bethe-Bloch stopping: {e}")

class ElectronIonCollision(CollisionProcess):
    """Electron–ion collisions using Spitzer frequencies.

    Only a simple drag term is applied; energy diffusion and large-angle
    scattering are not modelled."""
    def apply(self, solver):
        try:
            for name, spc in solver.species.items():
                if spc['q'] < 0:  # Electrons
                    ne = solver.field_manager.ne
                    Te = solver.field_manager.Te
                    νei = nu_ei_spitzer(ne, Te)
                    # Apply drag force (reduce velocity)
                    spc['vel'] -= νei[:, np.newaxis] * spc['vel'] * solver.dt
        except Exception as e:
            logger.error(f"Error applying electron-ion collisions: {e}")

class ElectronNeutralCollision(CollisionProcess):
    """Electron–neutral collisions with a constant cross-section.

    Assumes isotropic scattering and ignores energy dependence."""

    def __init__(self, sigma_en=1e-19):
        self.sigma_en = sigma_en

    def apply(self, solver):
        try:
            for name, spc in solver.species.items():
                if spc['q'] < 0:  # Electrons
                    ne = solver.field_manager.ne
                    Te = solver.field_manager.Te
                    nn = solver.field_manager.nn
                    νen = nu_en(ne, Te, nn, self.sigma_en)
                    # Apply drag force (reduce velocity)
                    spc['vel'] -= νen[:, np.newaxis] * spc['vel'] * solver.dt
        except Exception as e:
            logger.error(f"Error applying electron-neutral collisions: {e}")

class IonizationProcess(CollisionProcess):
    """Ionization of neutrals by electron impact.

    Particle creation is represented only by rate sampling; actual particle
    insertion is left as a placeholder."""
    def __init__(self, ionization_energy=13.6, cross_section_file="ionization_cross_section.h5"):
        self.ionization_energy = ionization_energy * e_charge  # Convert eV to Joules
        self.cross_section_data = CrossSectionData(cross_section_file)

    def apply(self, solver):
        try:
            for name, spc in solver.species.items():
                if spc['q'] < 0:  # Electrons
                    ne = solver.field_manager.ne
                    Te = solver.field_manager.Te
                    nn = solver.field_manager.nn
                    # Use energy-dependent cross-section
                    sigma_ion = self.cross_section_data(Te)
                    # Ionization rate
                    ionization_rate = ne * sigma_ion * np.sqrt(8 * kB * Te / (pi * m_e))
                    # Create new ions and electrons
                    num_new_ions = np.random.poisson(ionization_rate * nn * solver.dt)
                    # Add new particles (simplified - needs proper distribution)
                    # ... (implementation for adding new particles) ...
        except Exception as e:
            logger.error(f"Error applying ionization process: {e}")

class RecombinationProcess(CollisionProcess):
    """Radiative recombination of ions and electrons.

    Particle removal is not yet implemented and requires a proper selection
    mechanism."""
    def __init__(self, recombination_rate=1e-14):
        self.recombination_rate = recombination_rate

    def apply(self, solver):
        try:
            for name, spc in solver.species.items():
                if spc['q'] > 0:  # Ions
                    ne = solver.field_manager.ne
                    ni = solver.field_manager.ni
                    # Recombination rate
                    recombination_rate = self.recombination_rate * ne * ni
                    # Remove ions and electrons
                    num_removed_ions = np.random.poisson(recombination_rate * solver.dt)
                    # Remove particles (simplified - needs proper selection)
                    # ... (implementation for removing particles) ...
        except Exception as e:
            logger.error(f"Error applying recombination process: {e}")

# --------------------------------------
# D-D Fusion Reactions (simplified)
# --------------------------------------
class DDFusion(CollisionProcess):
    """Deuterium–Deuterium fusion reactions (simplified).

    Reaction products are not generated; only the rate is estimated."""
    def __init__(self, cross_section_file="dd_fusion_cross_section.h5"):
        self.cross_section_data = CrossSectionData(cross_section_file)

    def apply(self, solver):
        try:
            for name, spc in solver.species.items():
                if spc['q'] == e_charge and spc['m'] == 2 * m_p:  # Deuterium ions
                    # Use energy-dependent cross-section
                    sigma_fusion = self.cross_section_data(spc['energy'])
                    # Fusion rate
                    fusion_rate = spc['density'] * sigma_fusion * np.sqrt(8 * kB * spc['temperature'] / (pi * spc['m']))
                    # Create new particles (simplified - needs proper distribution)
                    num_new_neutrons = np.random.poisson(fusion_rate * solver.dt)
                    # ... (implementation for adding new neutrons) ...
        except Exception as e:
            logger.error(f"Error applying D-D fusion: {e}")

# --------------------------------------
# Braginskii Transport Coefficients
# --------------------------------------
@njit
def braginskii_coeffs(ne, Te, Bmag):
    """Computes Braginskii transport coefficients."""
    try:
        νei = nu_ei_spitzer(ne, Te)
        ωce = e_charge * Bmag / m_e
        x = ωce / (νei + 1e-30)
        κ_par = 3.16 * (kB**2 * ne * Te) / (m_e * (νei + 1e-30))
        κ_per = κ_par / (1 + x**2)
        return κ_par, κ_per
    except Exception as e:
        logger.error(f"Error computing Braginskii coefficients: {e}")
        return 0.0, 0.0

# --------------------------------------
# Main CollisionModel integrating all
# --------------------------------------
class CollisionModel(CollisionOperator):
    """Aggregate collision model for fluid simulations.

    Implements basic electron–ion relaxation, optional ionization and
    recombination rates, and utility routines for checkpointing.  The
    collisional–radiative network and PIC coupling are skeletal and subject
    to future expansion."""

    def __init__(self, config):
        self.config = config
        self.adas_file = config.get('adas_file', None)
        self.crn = CollisionalRadiativeNetwork(self.adas_file) if self.adas_file is not None else None
        self.checkpoint_data = {}
        self.accumulators = {}
        self.caches = {}
        # Load cross-section data
        self.ionization_cross_section = CrossSectionData(config.get('ionization_cross_section_file', "ionization_cross_section.h5"))
        self.dd_fusion_cross_section = CrossSectionData(config.get('dd_fusion_cross_section_file', "dd_fusion_cross_section.h5"))
        logger.info("CollisionModel initialized.")

    def apply(self, state: SimulationState, dt):
        try:
            rho = state.density  # m^-3
            ne = rho / m_p
            Te = state.electron_temperature
            Ti = state.ion_temperature
            nn = state.neutral_density if hasattr(state, 'neutral_density') else np.zeros_like(ne)
            # explicit PIC kernel example
            # threads / blocks
            nx, ny, nz = rho.shape
            threads = (8, 8, 8)
            blocks = ((nx + 7) // 8, (ny + 7) // 8, (nz + 7) // 8)
            # collision_gpu_kernel[blocks, threads](rho, Te, state.νei, state.νee) # Assuming state has νei and νee

            # electron-neutral collisions
            # state.νen = nu_en(ne, Te, nn) # Assuming state has νen

            # implicit relaxation
            νei = nu_ei_spitzer(ne, Te)
            self.caches['nu_ei'] = νei
            self.accumulators['steps'] = self.accumulators.get('steps', 0) + 1
            Te_new, Ti_new = relax_ei_implicit(Te, Ti, νei, dt)
            state.electron_temperature, state.ion_temperature = Te_new, Ti_new

            # anisotropy
            # if hasattr(state, 'Tpar'): # Assuming state has Tpar and Tper
            #     state.Tpar, state.Tper = relax_anisotropy(
            #         state.Tpar, state.Tper, nu_ii(ne, Ti), dt)

            # collisional-radiative
            if self.crn:
                ion_r, rec_r = self.crn.rates(Te, ne)  # m^3/s
                if hasattr(state, 'neutral_density'):
                    state.neutral_density -= ion_r * state.neutral_density * dt  # m^-3
                    if hasattr(state, 'ion_density'):
                        state.ion_density += (ion_r * state.neutral_density - rec_r * state.ion_density) * dt  # m^-3

            # ohmic heating J^2/sigma = eta*J^2
            J = state.field_manager.get_J()
            state.internal_energy += (νei * np.sum(J**2, axis=0) / np.maximum(rho, 1e-30)) * dt

            # diagnostics
            state.collision_diag = self.diagnostics(state)
        except Exception as e:
            logger.error(f"Error applying collision model: {e}")

    def diagnostics(self, state: SimulationState):
        v = state.velocity
        return {
            'm0': np.mean(v, axis=(0, 1, 2)),
            'm2': np.mean(v**2, axis=(0, 1, 2))
        }

    def pic_collision_handler(self):
        from warp_piclibrary import PICCollisionHandler
        return PICCollisionHandler(lambda ne, Te, Z=1.0: nu_ei_spitzer(ne, Te, Z))

    def checkpoint(self):
        self.checkpoint_data = {
            'ionization_cross_section': getattr(self.ionization_cross_section, 'to_dict', lambda: self.ionization_cross_section)(),
            'dd_fusion_cross_section': getattr(self.dd_fusion_cross_section, 'to_dict', lambda: self.dd_fusion_cross_section)(),
            'crn_state': self.crn,
            'accumulators': self.accumulators,
            'caches': self.caches,
            'random_state': np.random.get_state(),
        }
        return self.checkpoint_data

    def restart(self, data):
        """Restore the model state from ``data`` produced by :meth:`checkpoint`.

        Parameters
        ----------
        data: dict
            Dictionary produced by :meth:`checkpoint`.

        The method reloads cross‑section tables, cached values, accumulator
        counters and the global ``numpy`` random number generator state so that
        subsequent calls to :meth:`checkpoint` or stochastic routines reproduce
        the behaviour prior to checkpointing.
        """
        if not isinstance(data, dict):
            raise ValueError("Restart data must be a dictionary")

        # --- Cross‑section tables ---
        ion_data = data.get('ionization_cross_section')
        if isinstance(ion_data, dict):
            # Recreate ``CrossSectionData`` including its interpolation object
            self.ionization_cross_section = CrossSectionData.from_dict(ion_data)
        else:
            # Allow already‑constructed objects (useful for tests)
            self.ionization_cross_section = ion_data

        dd_data = data.get('dd_fusion_cross_section')
        if isinstance(dd_data, dict):
            self.dd_fusion_cross_section = CrossSectionData.from_dict(dd_data)
        else:
            self.dd_fusion_cross_section = dd_data

        # --- Internal state ---
        self.crn = data.get('crn_state')

        # Copy so that caller cannot mutate our internal dictionaries through
        # references obtained from ``data``.
        self.accumulators = dict(data.get('accumulators', {}))
        self.caches = dict(data.get('caches', {}))

        # --- RNG state ---
        rng_state = data.get('random_state')
        if rng_state is not None:
            try:
                np.random.set_state(rng_state)
            except Exception as e:
                logger.warning(f"Failed to restore RNG state: {e}")
                # Fall back to a deterministic seed so behaviour after restart
                # remains reproducible even if the state object is corrupt.
                np.random.seed(0)
        else:
            # No RNG state stored – reset to a deterministic seed rather than
            # continuing from an uncontrolled global state.
            np.random.seed(0)

        # Keep a copy of the checkpoint for idempotency checks
        self.checkpoint_data = dict(data)
