"""Collision model utilities for particle and fluid simulations.

Features
--------
* Coulomb logarithm with quantum diffraction corrections
* Spitzer collision frequencies including electron–neutral collisions
* Implicit electron–ion temperature relaxation
* Energy-dependent cross-section lookup tables
* Braginskii transport coefficient helper
* Simplified D–D fusion rate estimates
* Fokker–Planck velocity-space diffusion operator
* Anisotropy relaxation for directional temperatures
* Collisional–radiative network support
* Checkpoint and restart helpers

Future Work
-----------
* GPU acceleration and detailed diagnostics
"""

import numpy as np
import math
from scipy.interpolate import interp1d, RegularGridInterpolator
from numba import njit, prange, cuda
import logging
import types

try:  # optional dependency
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError("h5py is required; install dpf2[warpx]") from exc


from typing import List, Dict, Tuple, Optional

from .models import PhysicsModule
from .utils import SimulationState

logger = logging.getLogger("CollisionModel")
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
m_d = 3.34358377e-27  # Deuterium mass


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
    denom = 3 * (4 * pi * epsilon0) ** 2 * np.sqrt(m_e) * (kB * Te) ** 1.5
    return coef / (denom + 1e-30)


@njit
def nu_ee(ne, Te, lnL=10.0):
    return nu_ei_spitzer(ne, Te, lnL, Z=1.0) * np.sqrt(2)


@njit
def nu_ii(ni, Ti, lnL=10.0, Z=1.0, mi=m_p):
    coef = 4 * np.sqrt(pi) * ni * Z**4 * e_charge**4 * lnL
    denom = 3 * (4 * pi * epsilon0) ** 2 * np.sqrt(mi) * (kB * Ti) ** 1.5
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
            with h5py.File(filename, "r") as f:
                self.energy = f["energy"][:]
                self.cross_section = f["cross_section"][:]
            self.interp = interp1d(
                self.energy, self.cross_section, bounds_error=False, fill_value=0.0
            )
        except Exception as e:
            logger.error(f"Error loading cross-section data from {filename}: {e}")
            self.energy = []
            self.cross_section = []
            self.interp = lambda E: 0.0  # Default: zero cross-section

    def to_dict(self):
        """Return a serializable representation of the table."""
        return {
            "energy": list(self.energy),
            "cross_section": list(self.cross_section),
        }

    @classmethod
    def from_dict(cls, data):
        """Create a :class:`CrossSectionData` from checkpoint data."""
        obj = cls.__new__(cls)
        obj.energy = np.array(data.get("energy", []))
        obj.cross_section = np.array(data.get("cross_section", []))
        try:
            obj.interp = interp1d(
                obj.energy, obj.cross_section, bounds_error=False, fill_value=0.0
            )
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

    def __init__(
        self,
        name,
        Z_eff: int = 1,
        I_mean_ev: float = 13.6,
        speed_of_light: float = 299792458.0,
    ):

        self.name = name
        self.Z_eff = Z_eff
        self.I_mean = I_mean_ev * e_charge  # Convert eV to Joules
        # Allow speed of light to be provided, avoiding dependencies on PICSolver
        self.c = speed_of_light

    def apply(self, state: SimulationState, dt: float):
        """Applies the Bethe–Bloch stopping power to the ions."""
        try:
            if not hasattr(state, "species"):
                return
            for name, spc in state.species.items():
                if name == self.name:
                    vel = spc["vel"]
                    beta = np.linalg.norm(vel, axis=1) / self.c
                    gamma = 1.0 / np.sqrt(1.0 - beta**2)
                    ne = getattr(
                        getattr(state, "field_manager", types.SimpleNamespace(ne=0)),
                        "ne",
                        0,
                    )
                    ne = np.broadcast_to(ne, beta.shape)
                    stopping_power = (
                        4 * pi * self.Z_eff**2 * e_charge**4 * ne / (m_e * self.c**2)
                    ) * (
                        np.log((2 * m_e * self.c**2 * beta**2 * gamma**2) / self.I_mean)
                        - beta**2
                    )
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

    def apply(self, state: SimulationState, dt: float):
        try:
            if not hasattr(state, "species"):
                return
            for name, spc in state.species.items():
                if spc["q"] < 0:  # Electrons
                    ne = state.field_manager.ne
                    Te = state.field_manager.Te
                    νei = nu_ei_spitzer(ne, Te)
                    # Apply drag force (reduce velocity)
                    spc["vel"] -= νei[:, np.newaxis] * spc["vel"] * dt
        except Exception as e:
            logger.error(f"Error applying electron-ion collisions: {e}")


class ElectronNeutralCollision(CollisionProcess):
    """Electron–neutral collisions with a constant cross-section.

    Assumes isotropic scattering and ignores energy dependence."""

    def __init__(self, sigma_en=1e-19):
        self.sigma_en = sigma_en

    def apply(self, state: SimulationState, dt: float):
        try:
            if not hasattr(state, "species"):
                return
            for name, spc in state.species.items():
                if spc["q"] < 0:  # Electrons
                    ne = state.field_manager.ne
                    Te = state.field_manager.Te
                    nn = state.field_manager.nn
                    νen = nu_en(ne, Te, nn, self.sigma_en)
                    # Apply drag force (reduce velocity)
                    spc["vel"] -= νen[:, np.newaxis] * spc["vel"] * dt
        except Exception as e:
            logger.error(f"Error applying electron-neutral collisions: {e}")


class IonizationProcess(CollisionProcess):
    """Ionization of neutrals by electron impact.

    Particle creation is represented by sampling from the ionization rate and
    inserting new ions and electrons into the simulation state."""

    def __init__(
        self, ionization_energy=13.6, cross_section_file="ionization_cross_section.h5"
    ):
        self.ionization_energy = ionization_energy * e_charge  # Convert eV to Joules
        self.cross_section_data = CrossSectionData(cross_section_file)

    def apply(self, state: SimulationState, dt: float):
        try:
            if not hasattr(state, "species"):
                return
            if not hasattr(state, "field_manager"):
                return
            ne = state.field_manager.ne
            Te = state.field_manager.Te
            nn = getattr(state.field_manager, "nn", 0.0)
            sigma_ion = self.cross_section_data(Te)

            ion_rate = ne * sigma_ion * np.sqrt(8 * kB * Te / (pi * m_e))

            def _mean(val):
                try:
                    return float(np.mean(val))
                except Exception:
                    try:
                        return float(sum(val) / len(val))
                    except Exception:
                        return float(val)

            lam = _mean(ion_rate * nn) * dt
            num_pairs = np.random.poisson(lam)
            if num_pairs <= 0:
                return
            positions = state.sample_positions(num_pairs)
            Te_mean = _mean(Te)
            vel_e = state.sample_velocities(num_pairs, Te_mean, m_e)
            vel_i = state.sample_velocities(num_pairs, Te_mean, m_p)
            state.add_particles("e", -e_charge, m_e, positions, vel_e)
            state.add_particles("ion", e_charge, m_p, positions, vel_i)
        except Exception as e:
            logger.error(f"Error applying ionization process: {e}")


class RecombinationProcess(CollisionProcess):
    """Radiative recombination of ions and electrons.

    Particle removal removes ion–electron pairs using a simple momentum
    matching strategy to conserve charge and approximately conserve momentum."""

    def __init__(self, recombination_rate=1e-14):
        self.recombination_rate = recombination_rate

    def apply(self, state: SimulationState, dt: float):
        try:
            if not hasattr(state, "species"):
                return
            if not hasattr(state, "field_manager"):
                return
            ne = state.field_manager.ne
            ni = getattr(state.field_manager, "ni", ne)

            def _mean(val):
                try:
                    return float(np.mean(val))
                except Exception:
                    try:
                        return float(sum(val) / len(val))
                    except Exception:
                        return float(val)

            lam = _mean(self.recombination_rate * ne * ni) * dt
            num_pairs = np.random.poisson(lam)
            # identify electron and ion species
            e_name = next(
                (n for n, sp in state.species.items() if sp.get("q", 0.0) < 0), None
            )
            i_name = next(
                (n for n, sp in state.species.items() if sp.get("q", 0.0) > 0), None
            )
            if e_name is None or i_name is None:
                return
            sp_e = state.species[e_name]
            sp_i = state.species[i_name]
            num_pairs = min(num_pairs, sp_e["pos"].shape[0], sp_i["pos"].shape[0])
            for _ in range(num_pairs):
                if sp_e["pos"].shape[0] == 0 or sp_i["pos"].shape[0] == 0:
                    break
                if hasattr(np.random, "randint"):
                    idx_e = int(np.random.randint(0, sp_e["pos"].shape[0]))
                else:
                    idx_e = 0
                p_e = sp_e["m"] * sp_e["vel"][idx_e]
                ion_mom = sp_i["m"] * sp_i["vel"]
                try:
                    idx_i = int(np.argmin(np.sum((ion_mom + p_e) ** 2, axis=1)))
                except Exception:
                    idx_i = 0
                state.remove_particles(e_name, [idx_e])
                state.remove_particles(i_name, [idx_i])
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

    def apply(self, state: SimulationState, dt: float):
        try:
            if not hasattr(state, "species"):
                return
            for name, spc in state.species.items():
                if spc["q"] == e_charge and spc["m"] == 2 * m_p:  # Deuterium ions
                    # Use energy-dependent cross-section
                    sigma_fusion = self.cross_section_data(spc["energy"])
                    # Fusion rate
                    fusion_rate = (
                        spc["density"]
                        * sigma_fusion
                        * np.sqrt(8 * kB * spc["temperature"] / (pi * spc["m"]))
                    )
                    # Create new particles (simplified - needs proper distribution)
                    num_new_neutrons = np.random.poisson(fusion_rate * dt)
                    # ... (implementation for adding new neutrons) ...
        except Exception as e:
            logger.error(f"Error applying D-D fusion: {e}")


# --------------------------------------
# Fokker–Planck and anisotropy operators
# --------------------------------------
class FokkerPlanckOperator(CollisionProcess):
    """Simple Fokker–Planck velocity-space diffusion.

    The operator applies an isotropic diffusion to particle velocities.  A
    linear drag term may also be supplied.  The implementation is intentionally
    lightweight and meant for regression tests rather than production use."""

    def __init__(self, diffusion_coeff: float = 1e-3, drag_coeff: float = 0.0):
        self.diffusion_coeff = diffusion_coeff
        self.drag_coeff = drag_coeff

    def apply(self, state: SimulationState, dt: float):
        try:
            if not hasattr(state, "species"):
                return
            for spc in state.species.values():
                v = spc.get("vel")
                if v is None:
                    continue
                noise = np.random.normal(0.0, 1.0, size=v.shape)
                spc["vel"] = (1.0 - self.drag_coeff * dt) * v + np.sqrt(
                    2.0 * self.diffusion_coeff * dt
                ) * noise
        except Exception as e:
            logger.error(f"Error applying Fokker-Planck operator: {e}")


class AnisotropyRelaxation(CollisionProcess):
    """Relax anisotropy in the velocity distribution."""

    def __init__(self, rate: float = 1.0):
        self.rate = rate

    def apply(self, state: SimulationState, dt: float):
        try:
            if not hasattr(state, "species"):
                return
            for spc in state.species.values():
                v = spc.get("vel")
                if v is None or len(v) == 0:
                    continue
                data = v.data if hasattr(v, "data") else v
                n = len(data)
                sums = [0.0, 0.0, 0.0]
                sqs = [0.0, 0.0, 0.0]
                for row in data:
                    for j in range(3):
                        val = row[j]
                        sums[j] += val
                        sqs[j] += val * val
                var = [sqs[j] / n - (sums[j] / n) ** 2 for j in range(3)]
                if all(vv > 0.0 for vv in var):
                    mean_var = sum(var) / 3.0
                    scale = [(mean_var / (var[j] + 1e-30)) ** 0.5 for j in range(3)]
                    for row in data:
                        for j in range(3):
                            row[j] *= (1.0 - self.rate * dt) + self.rate * dt * scale[j]
                    if hasattr(v, "data"):
                        spc["vel"] = np.array(data)
        except Exception as e:
            logger.error(f"Error applying anisotropy relaxation: {e}")


class CollisionalRadiativeNetwork(CollisionProcess):
    """Very small collisional–radiative network model.

    Parameters
    ----------
    levels:
        Optional list of level names.  If omitted an empty network is created.
    coll_rates, rad_rates:
        Dictionaries mapping ``(i, j)`` level index pairs to rate coefficients
        in ``s⁻¹``.  Collisional rates represent transitions due to particle
        encounters while radiative rates model spontaneous emission.
    filename:
        When provided, the class attempts to load ``levels``, ``coll_rates`` and
        ``rad_rates`` datasets from an HDF5 file.  Missing data results in empty
        tables and a warning.
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        *,
        levels: Optional[List[str]] = None,
        coll_rates: Optional[Dict[Tuple[int, int], float]] = None,
        rad_rates: Optional[Dict[Tuple[int, int], float]] = None,
    ):
        self.levels = levels or []
        self.coll_rates = coll_rates or {}
        self.rad_rates = rad_rates or {}
        if filename is not None:
            self._load_file(filename)

    def _load_file(self, filename):
        try:
            with h5py.File(filename, "r") as f:
                self.levels = [str(x) for x in f.get("levels", [])]
                self.coll_rates = {
                    (int(i), int(j)): float(r) for i, j, r in f.get("coll_rates", [])
                }
                self.rad_rates = {
                    (int(i), int(j)): float(r) for i, j, r in f.get("rad_rates", [])
                }
        except Exception as e:  # pragma: no cover - file parsing errors
            logger.warning(f"Failed to load collisional–radiative data: {e}")

    def apply(self, state: SimulationState, dt: float):
        try:
            pops = getattr(state, "radiation", {}).get("populations")
            if pops is None:
                return
            # convert to a mutable python list of floats
            if hasattr(pops, "data"):
                pops_list = [float(x) for x in pops.data]
            else:
                pops_list = [float(x) for x in pops]
            for (i, j), rate in self.coll_rates.items():
                delta = rate * pops_list[i] * dt
                pops_list[i] -= delta
                pops_list[j] += delta
            for (i, j), rate in self.rad_rates.items():
                delta = rate * pops_list[i] * dt
                pops_list[i] -= delta
                pops_list[j] += delta
            state.radiation["populations"] = pops_list
        except Exception as e:
            logger.error(f"Error applying collisional–radiative network: {e}")

    # Simple aggregated ionisation/recombination rate helpers used by
    # ``CollisionModel``.  These return average rates over all transitions.
    def rates(self, Te, ne):
        ion_r = sum(self.coll_rates.values())
        rec_r = sum(self.rad_rates.values())
        return ion_r, rec_r


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

    def __init__(self, config, processes: Optional[List[CollisionProcess]] = None):
        self.config = config
        self.adas_file = config.get("adas_file", None)
        self.crn = (
            CollisionalRadiativeNetwork(self.adas_file)
            if self.adas_file is not None
            else None
        )
        self.checkpoint_data = {}
        self.accumulators = {}
        self.caches = {}
        # Optional list of additional collision processes
        self.processes: List[CollisionProcess] = processes or []
        # Load cross-section data
        self.ionization_cross_section = CrossSectionData(
            config.get("ionization_cross_section_file", "ionization_cross_section.h5")
        )
        self.dd_fusion_cross_section = CrossSectionData(
            config.get("dd_fusion_cross_section_file", "dd_fusion_cross_section.h5")
        )
        logger.info("CollisionModel initialized.")

    def apply(self, state: SimulationState, dt):
        try:
            for process in self.processes:
                process.apply(state, dt)

            rho = state.density  # m^-3
            ne = rho / m_p
            Te = state.electron_temperature
            Ti = state.ion_temperature
            nn = (
                state.neutral_density
                if hasattr(state, "neutral_density")
                else np.zeros_like(ne)
            )
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
            self.caches["nu_ei"] = νei
            self.accumulators["steps"] = self.accumulators.get("steps", 0) + 1
            Te_new, Ti_new = relax_ei_implicit(Te, Ti, νei, dt)
            state.electron_temperature, state.ion_temperature = Te_new, Ti_new

            # anisotropy
            # if hasattr(state, 'Tpar'): # Assuming state has Tpar and Tper
            #     state.Tpar, state.Tper = relax_anisotropy(
            #         state.Tpar, state.Tper, nu_ii(ne, Ti), dt)

            # collisional-radiative
            if self.crn:
                ion_r, rec_r = self.crn.rates(Te, ne)  # m^3/s
                if hasattr(state, "neutral_density"):
                    state.neutral_density -= ion_r * state.neutral_density * dt  # m^-3
                    if hasattr(state, "ion_density"):
                        state.ion_density += (
                            ion_r * state.neutral_density - rec_r * state.ion_density
                        ) * dt  # m^-3

            # ohmic heating J^2/sigma = eta*J^2
            J = state.field_manager.get_J()
            state.internal_energy += (
                νei * np.sum(J**2, axis=0) / np.maximum(rho, 1e-30)
            ) * dt

            # diagnostics
            state.collision_diag = self.diagnostics(state)
        except Exception as e:
            logger.error(f"Error applying collision model: {e}")

    def diagnostics(self, state: SimulationState):
        v = state.velocity
        return {"m0": np.mean(v, axis=(0, 1, 2)), "m2": np.mean(v**2, axis=(0, 1, 2))}

    def pic_collision_handler(self):
        from warp_piclibrary import PICCollisionHandler

        return PICCollisionHandler(lambda ne, Te, Z=1.0: nu_ei_spitzer(ne, Te, Z))

    def checkpoint(self):
        self.checkpoint_data = {
            "ionization_cross_section": getattr(
                self.ionization_cross_section,
                "to_dict",
                lambda: self.ionization_cross_section,
            )(),
            "dd_fusion_cross_section": getattr(
                self.dd_fusion_cross_section,
                "to_dict",
                lambda: self.dd_fusion_cross_section,
            )(),
            "crn_state": self.crn,
            "accumulators": self.accumulators,
            "caches": self.caches,
            "random_state": np.random.get_state(),
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
        ion_data = data.get("ionization_cross_section")
        if isinstance(ion_data, dict):
            # Recreate ``CrossSectionData`` including its interpolation object
            self.ionization_cross_section = CrossSectionData.from_dict(ion_data)
        else:
            # Allow already‑constructed objects (useful for tests)
            self.ionization_cross_section = ion_data

        dd_data = data.get("dd_fusion_cross_section")
        if isinstance(dd_data, dict):
            self.dd_fusion_cross_section = CrossSectionData.from_dict(dd_data)
        else:
            self.dd_fusion_cross_section = dd_data

        # --- Internal state ---
        self.crn = data.get("crn_state")

        # Copy so that caller cannot mutate our internal dictionaries through
        # references obtained from ``data``.
        self.accumulators = dict(data.get("accumulators", {}))
        self.caches = dict(data.get("caches", {}))

        # --- RNG state ---
        rng_state = data.get("random_state")
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
