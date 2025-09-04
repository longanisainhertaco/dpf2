# warpx_wrapper.py

"""Lightweight WarpX wrapper.

This module provides grid construction, solver setup and particle management
with optional collision handling.  It also includes particle energy and
phase‑space diagnostics, simple fluid–PIC interpolation, region- and
species‑specific collision control and runtime parameter updates.  Additional
boundary condition types, optional relativistic corrections, a rudimentary
quantum emission hook and time‑dependent boundary fields aim to improve
robustness of the wrapper in minimal environments.
"""

import os
import time
import threading
import queue
import socket
import json
import numpy as np
import logging
try:  # optional dependency
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "h5py is required; install dpf2[warpx]"
    ) from exc
try:  # optional dependency
    import picmi
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "picmi is required; install dpf2[warpx]"
    ) from exc
try:  # optional dependency
    import adios2
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "adios2 is required; install dpf2[warpx]"
    ) from exc
try:  # optional dependency
    import amrex
    from amrex import EBIndexSpace
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "amrex is required; install dpf2[warpx]"
    ) from exc
from .collision_model import CollisionModel  # Assuming you have this
from .utils import FieldManager
from ..core.bases import CouplingState


def _resample_array(arr, new_shape, method="linear"):
    """Resample ``arr`` to ``new_shape`` using a simple separable scheme.

    Parameters
    ----------
    arr: :class:`numpy.ndarray`
        Input array with up to three dimensions.
    new_shape: tuple
        Desired output shape.
    method: str
        ``"nearest"`` or ``"linear"`` interpolation.  ``"linear"``
        performs a trilinear interpolation by applying :func:`numpy.interp`
        sequentially along each axis.

    Returns
    -------
    :class:`numpy.ndarray`
        Resampled array.
    """

    try:
        if arr.shape == tuple(new_shape):
            return np.array(arr)

        res = np.array(arr, dtype=float)
        for axis, n_new in enumerate(new_shape):
            n_old = res.shape[axis]
            if n_old == n_new:
                continue
            old_grid = np.linspace(0.0, 1.0, n_old)
            new_grid = np.linspace(0.0, 1.0, n_new)
            res = np.moveaxis(res, axis, 0)
            if method == "nearest":
                idx = np.clip(np.round(new_grid * (n_old - 1)).astype(int), 0, n_old - 1)
                res = res[idx]
            else:  # linear
                tmp = np.empty((n_new,) + res.shape[1:], dtype=res.dtype)
                for idx in np.ndindex(res.shape[1:]):
                    tmp[(slice(None),) + idx] = np.interp(new_grid, old_grid, res[(slice(None),) + idx])
                res = tmp
            res = np.moveaxis(res, 0, axis)
        return res
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.error(f"Error during array resampling: {exc}")
        raise

# Configure logger
logger = logging.getLogger('WarpXWrapper')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# Physical constants
c = 299792458.0
epsilon0 = 8.854187817e-12
mu0 = 4 * np.pi * 1e-7
kB = 1.380649e-23
e_charge = 1.602176634e-19
m_e = 9.10938356e-31
pi = np.pi


def _compute_kinetic_energy(vel, mass, relativistic=False):
    """Compute kinetic energy for ``vel`` with optional relativistic correction."""
    v = np.asarray(vel)
    v2 = np.sum(v ** 2, axis=1)
    if not relativistic:
        return 0.5 * mass * v2
    energies = []
    for vv in v2:
        beta2 = float(vv) / c ** 2
        beta2 = min(max(beta2, 0.0), 0.999999)
        gamma = 1.0 / (1.0 - beta2) ** 0.5
        energies.append((gamma - 1.0) * mass * c ** 2)
    return np.array(energies)

class Field:
    """
    Thin wrapper for field and particle arrays, matching fluid solver style.
    """
    def __init__(self, name, array, spacing):
        self.name = name
        self.array = array
        self.spacing = spacing

class WarpXWrapper:
    """Simplified interface to WarpX for grid setup and particle integration.

    Limitations:
        - Relativistic and quantum effects are not fully modeled.
        - Diagnostics and boundary options are restricted.
    """

    def __init__(self,
                 domain_lower, domain_upper,
                 grid_shape, dx,
                 species_params,
                 pic_params,
                 performance,
                 fluid_callback=None,
                 unity_params=None,
                 circuit=None,
                 field_manager: FieldManager = None): # Add field_manager
        # Parameter validation and default values
        required = ['FN_A', 'FN_B', 'surface_cells', 'geometry', 'boundary',
                    'particle_shape', 'adios_engine', 'adios_file', 'pusher']
        for k in required:
            if k not in pic_params:
                raise KeyError(f"Missing required pic_params key: '{k}'")

        self.domain_lower = tuple(domain_lower)
        self.domain_upper = tuple(domain_upper)
        self.grid_shape = tuple(grid_shape)
        self.dx = float(dx)
        self.fluid_callback = fluid_callback
        self.unity_params = unity_params or {}
        self.circuit = circuit
        self.config = pic_params
        self.field_manager = field_manager # Store FieldManager

        self.relativistic_corrections = pic_params.get('relativistic_corrections', False)
        self.quantum_emission = pic_params.get('quantum_emission', False)
        self.time_dependent_boundaries = pic_params.get('time_dependent_boundaries', False)

        # Fowler–Nordheim parameters
        self.fn_A = pic_params['FN_A']
        self.fn_B = pic_params['FN_B']
        self.surface_cells = pic_params['surface_cells']

        # Default parameters
        self.pml_ncell = pic_params.get('pml_ncell', 10)
        self.pml_sigma_max = pic_params.get('pml_sigma_max', 1.0)
        self.maxwell_order = pic_params.get('maxwell_order', 4)
        self.cfl = pic_params.get('cfl', 0.99)
        self.pusher_cfl = pic_params.get('pusher_cfl', 0.9)
        self.collision_algorithm = pic_params.get('collision_algorithm', 'Nanbu')
        self.collision_enabled = pic_params.get('collision', False)
        self.collision_configs = pic_params.get('collisions', [])
        self.interp_method = pic_params.get('interp_method', 'linear')
        self.diagnostic_bins = pic_params.get('diagnostic_bins', 50)

        # Build grid
        geom_type = pic_params['geometry']
        try:
            if geom_type == 'rz':
                self.grid = picmi.CylindricalGrid(
                    number_of_cells_r=grid_shape[0],
                    number_of_cells_z=grid_shape[1],
                    lower_bound_r=0.0,
                    upper_bound_r=domain_upper[0],
                    lower_bound_z=domain_lower[2],
                    upper_bound_z=domain_upper[2],
                    is_periodic_z=True
                )
            elif geom_type == 'cartesian':
                bc_cfg = pic_params.get('boundary_conditions', {})
                lower_bc = [
                    bc_cfg.get('x_lo', 'periodic'),
                    bc_cfg.get('y_lo', 'periodic'),
                    bc_cfg.get('z_lo', 'periodic'),
                ]
                upper_bc = [
                    bc_cfg.get('x_hi', 'periodic'),
                    bc_cfg.get('y_hi', 'periodic'),
                    bc_cfg.get('z_hi', 'periodic'),
                ]
                is_periodic = [
                    l == 'periodic' and u == 'periodic'
                    for l, u in zip(lower_bc, upper_bc)
                ]
                self.grid = picmi.Cartesian3DGrid(
                    number_of_cells=grid_shape,
                    lower_bound=domain_lower,
                    upper_bound=domain_upper,
                    lower_boundary_conditions=lower_bc,
                    upper_boundary_conditions=upper_bc,
                    is_periodic=is_periodic
                )
            else:
                raise ValueError(f"Invalid geometry type: {geom_type}")
        except Exception as e:
            logger.error(f"Error creating grid: {e}")
            raise

        # Maxwell solver
        try:
            pml_ncell = self.pml_ncell if pic_params['boundary'] == 'pml' else 0
            warpx_cfg = {'warpx.do_mpi': True, 'warpx.max_blocks': performance.get('gpu_block', 256),
                         'common.profiling': True}
            self.maxwell = picmi.MaxwellSolver(
                grid=self.grid,
                method='Yee', order=self.maxwell_order,
                cfl=self.cfl,
                pml_ncell=pml_ncell,
                warpx_config=warpx_cfg,
                use_cuda=performance.get('gpu', False)
            )
        except Exception as e:
            logger.error(f"Error creating Maxwell solver: {e}")
            raise

        # Pusher & Deposition
        try:
            pusher_type = pic_params['pusher']
            self.pusher = picmi.Pusher(method=pusher_type, cfl=self.pusher_cfl)
            self.deposition = picmi.Deposition(n_cells_same_as_grid=True,
                                              deposition_order=pic_params['particle_shape'])
        except Exception as e:
            logger.error(f"Error creating pusher or deposition: {e}")
            raise

        # Initialize WarpX
        try:
            self.warp = picmi.WarpX(solver=self.maxwell,
                                    pusher=self.pusher,
                                    deposition=self.deposition)
        except Exception as e:
            logger.error(f"Error initializing WarpX: {e}")
            raise

        # Embedded-boundary via AMReX
        if 'eb_geometry' in pic_params:
            try:
                eb_cfg = pic_params['eb_geometry']
                eb = EBIndexSpace()
                eb.defineGeometryFromSTL(eb_cfg['file'], origin=tuple(eb_cfg.get('origin', (0, 0, 0))),
                                         dx=(self.dx, self.dx, self.dx))
                self.warp.setEmbeddedBoundary(eb)
                logger.info("Embedded-boundary geometry applied from %s", eb_cfg['file'])
            except Exception as e:
                logger.error(f"Error setting up embedded boundary: {e}")
                raise

        # Species
        self.species = {}
        for sp in species_params:
            try:
                name = sp['name']
                dist = picmi.UniformDistribution(density=sp['density'] * sp.get('density_frac', 1.0))
                part = picmi.ParticleBinning(grid=self.grid,
                                             name=name,
                                             lower_bound=self.domain_lower,
                                             upper_bound=self.domain_upper,
                                             n_macroparticles_per_cell=sp.get('macro_per_cell', 1),
                                             particle_shape=pic_params['particle_shape'],
                                             movement_color=0)
                self.species[name] = sp
                self.warp.add_species(part,
                                      mass=sp['mass'],
                                      charge=sp['charge'],
                                      initial_distribution=dist)
            except Exception as e:
                logger.error(f"Error adding species {name}: {e}")
                raise

        # Collisions
        self.collision_models = []
        if self.collision_enabled:
            try:
                self.warp.enable_collisions(
                    algorithm=self.collision_algorithm, coulomb_log='dynamic'
                )
                for cfg in self.collision_configs:
                    if not cfg.get('enabled', True):
                        continue
                    sp1, sp2 = cfg.get('species', (None, None))
                    region = cfg.get('region')
                    if not sp1 or not sp2:
                        logger.warning(
                            "Collision config missing species pair: %s", cfg
                        )
                        continue
                    self.warp.add_collision(sp1, sp2, region=region)
                    model_cfg = cfg.get('model', {})
                    try:
                        self.collision_models.append(CollisionModel(model_cfg))
                    except Exception as exc:
                        logger.warning(
                            "Failed to initialise CollisionModel for %s-%s: %s",
                            sp1,
                            sp2,
                            exc,
                        )
            except Exception as e:
                logger.error(f"Error enabling collisions: {e}")
                raise

        # Initialize run
        try:
            self.warp.init_run()
        except Exception as e:
            logger.error(f"Error initializing WarpX run: {e}")
            raise

        # Sheath emission via Fowler–Nordheim
        try:
            E_surf = self.warp.GetInstalledField('E', 'surface')
            J_FN = self.fn_A * E_surf ** 2 * np.exp(-self.fn_B / E_surf)
            self.warp.add_particles('electron', region=self.surface_cells, current=J_FN)
        except Exception as e:
            logger.error(f"Error setting up Fowler-Nordheim emission: {e}")
            raise

        # ADIOS2 for field I/O
        try:
            self.adios = adios2.ADIOS()
            io = self.adios.DeclareIO("WarpXWrapperIO")
            io.SetEngine(pic_params['adios_engine'])
            for k, v in pic_params.get('adios_parameters', {}).items():
                io.SetParameter(k, str(v))
            # Define field variables
            self.field_vars = {}
            for comp in ['rho', 'Jx', 'Jy', 'Jz', 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
                arr = np.array(self.warp.get_field(comp))
                self.field_vars[comp] = io.DefineVariable(comp,
                                                          arr,
                                                          adios2.Shape(arr.shape),
                                                          adios2.Start([0] * arr.ndim),
                                                          adios2.Count(list(arr.shape)))
            # Define particle variables
            self.particle_vars = {}
            for name, sp in self.species.items():
                pts = self.warp.get_particle_container(name)
                pos = np.array(pts.get_positions())
                vel = np.array(pts.get_velocities())
                self.particle_vars[f"{name}_pos"] = io.DefineVariable(f"{name}_pos",
                                                                      pos, adios2.Shape(pos.shape),
                                                                      adios2.Start([0, 0]),
                                                                      adios2.Count(list(pos.shape)))
                self.particle_vars[f"{name}_vel"] = io.DefineVariable(f"{name}_vel",
                                                                      vel, adios2.Shape(vel.shape),
                                                                      adios2.Start([0, 0]),
                                                                      adios2.Count(list(vel.shape)))
            self.writer = io.Open(pic_params['adios_file'], adios2.Mode.Write)
        except Exception as e:
            logger.error(f"Error setting up ADIOS2 I/O: {e}")
            raise

        # Unity streaming thread
        self._queue = queue.Queue()
        if unity_params:
            try:
                self._u_thread = threading.Thread(target=self._unity_stream_thread)
                self._u_thread.daemon = True
                self._u_thread.start()
            except Exception as e:
                logger.error(f"Error starting Unity streaming thread: {e}")
                raise
        logger.info("WarpXWrapper initialized successfully.")

    def inject_boundary_fields(self):
        if not self.fluid_callback:
            return
        try:
            t = self.warp.get_time() if (self.time_dependent_boundaries and hasattr(self.warp, 'get_time')) else 0.0
            bnd = self.fluid_callback().get('boundary_fields', {})
            for comp, fld in bnd.items():
                if callable(fld):
                    raw = fld(t)
                elif isinstance(fld, dict) and 'times' in fld and 'values' in fld:
                    times = np.asarray(fld['times'])
                    values = np.asarray(fld['values'])
                    idx = np.searchsorted(times, t, side='right') - 1
                    idx = np.clip(idx, 0, len(times) - 1)
                    raw = values[idx]
                else:
                    raw = fld
                arr = _resample_array(np.array(raw), self.grid_shape, self.interp_method)
                data = arr.tolist() if hasattr(arr, 'tolist') else arr
                self.warp.set_boundary_field(data, comp)
        except Exception as e:
            logger.error(f"Error injecting boundary fields: {e}")

    def map_pic_to_fluid(self):
        if not self.fluid_callback or not self.field_manager:
            return
        try:
            rho_pic = np.array(self.warp.get_field('rho'))
            J_pic = np.stack(
                [np.array(self.warp.get_field(c)) for c in ('Jx', 'Jy', 'Jz')],
                axis=0,
            )
            rho_res = _resample_array(
                rho_pic, self.field_manager.rho.shape, self.interp_method
            )
            J_res = np.stack(
                [
                    _resample_array(
                        J_pic[i], self.field_manager.J.shape[1:], self.interp_method
                    )
                    for i in range(3)
                ],
                axis=0,
            )
            self.field_manager.deposit_charge(rho_res)
            self.field_manager.deposit_current(J_res)
        except Exception as e:
            logger.error(f"Error mapping PIC data to fluid: {e}")

    def _apply_relativistic_corrections(self, container):
        """Apply a simple relativistic gamma correction to particle velocities."""
        if not self.relativistic_corrections:
            return
        try:
            vel = np.array(container.get_velocities())
            v2 = np.sum(vel ** 2, axis=1)
            gamma = 1.0 / np.sqrt(1.0 - np.clip(v2 / c ** 2, 0.0, 0.999999))
            container.set_velocities((vel * gamma[:, None]).tolist())
        except Exception as exc:
            logger.error(f"Error applying relativistic correction: {exc}")

    def _handle_quantum_emission(self):
        """Very crude quantum emission placeholder."""
        if not self.quantum_emission:
            return
        try:
            Ez = np.array(self.warp.get_field('Ez'))
            chi = np.abs(Ez) / (m_e * c ** 2 / e_charge)
            prob = np.clip(chi ** 2, 0.0, 1.0)
            self._last_emission_prob = float(np.mean(prob))
        except Exception as exc:
            logger.error(f"Error in quantum emission module: {exc}")

    def record_diagnostics(self):
        """Records basic diagnostics from WarpX."""
        try:
            diagnostics = {}
            for name, sp in self.species.items():
                cont = self.warp.get_particle_container(name)
                vel = np.array(cont.get_velocities())
                pos = np.array(cont.get_positions())
                mass = sp['mass']
                energy = _compute_kinetic_energy(vel, mass, self.relativistic_corrections)
                diagnostics[f"{name}_kinetic_energy"] = float(np.sum(energy))
                hist, _ = np.histogram(energy, bins=self.diagnostic_bins)
                diagnostics[f"{name}_energy_hist"] = hist.tolist()
                phase, _, _ = np.histogram2d(
                    pos[:, 2], vel[:, 2], bins=self.diagnostic_bins
                )
                diagnostics[f"{name}_phase_space"] = phase.tolist()

            E = np.stack(
                [self.warp.get_field(c) for c in ('Ex', 'Ey', 'Ez')], axis=-1
            )
            fe = 0.5 * epsilon0 * np.sum(E ** 2) * self.dx ** 3
            diagnostics["field_energy"] = float(fe)
            logger.debug("Recorded diagnostics: %s", list(diagnostics.keys()))
            return diagnostics
        except Exception as e:
            logger.error(f"Error recording diagnostics: {e}")
            return {}

    def update_runtime_parameters(self, params):
        """Update runtime parameters for the PIC run.

        Parameters
        ----------
        params: dict
            Mapping of parameter names to new values.  Known keys are stored
            locally and forwarded to WarpX when possible.
        """

        for key, value in params.items():
            try:
                setattr(self, key, value)
                self.config[key] = value
                if hasattr(self.warp, "set_runtime_parameter"):
                    self.warp.set_runtime_parameter(key, value)
                logger.info("Runtime parameter %s updated to %s", key, value)
            except Exception as exc:
                logger.error("Error updating runtime parameter %s: %s", key, exc)

    def step(self, rho, J, E, B, dt):
        try:
            # Determine fluid_dt
            sub_dt = self.warp.get_current_time_step()
            subcycles = max(1, int(round(dt / sub_dt)))
            t0 = time.perf_counter()
            timings = {'inject': 0, 'pic': 0, 'circuit': 0, 'map': 0, 'diag': 0}
            for _ in range(subcycles):
                t_i = time.perf_counter()
                self.inject_boundary_fields()
                timings['inject'] += time.perf_counter() - t_i
                t_p = time.perf_counter()
                self.warp.step(1)
                for name in self.species:
                    self._apply_relativistic_corrections(self.warp.get_particle_container(name))
                self._handle_quantum_emission()
                timings['pic'] += time.perf_counter() - t_p
                if self.circuit:
                    t_c = time.perf_counter()
                    I_pic = self.get_total_current()
                    # The circuit interface expects the present state and an
                    # applied back‑EMF.  The stub circuit used here does not
                    # provide a detailed model so we pass the PIC current and
                    # zero EMF/inductance.
                    voltage = (
                        self.circuit.get_voltage()
                        if hasattr(self.circuit, "get_voltage")
                        else 0.0
                    )
                    state = CouplingState(current=I_pic, voltage=voltage)
                    self.circuit.step(state, 0.0, dt)
                    self.warp.applyBoundaryB(self.circuit.I)
                    timings['circuit'] += time.perf_counter() - t_c
            t_map = time.perf_counter()
            self.map_pic_to_fluid()
            timings['map'] += time.perf_counter() - t_map
            t_d = time.perf_counter()
            self.record_diagnostics()
            timings['diag'] += time.perf_counter() - t_d
            total = time.perf_counter() - t0
            npart = sum(len(self.warp.get_particle_container(n).get_positions()) for n in self.species)
            logger.info(f"Step: subcycles={subcycles}, total_time={total:.3f}s, particles={npart}")
            logger.debug(f"Timings: {timings}")

            # ADIOS2 checkpoint
            self.writer.BeginStep()
            for comp, var in self.field_vars.items():
                arr = np.array(self.warp.get_field(comp))
                self.writer.Put(var, arr)
            for key, var in self.particle_vars.items():
                name, typ = key.rsplit('_', 1)
                pts = self.warp.get_particle_container(name)
                data = np.array(pts.get_positions()) if typ == 'pos' else np.array(pts.get_velocities())
                self.writer.Put(var, data)
            self.writer.EndStep()
            return self.field_manager.get_E(), self.field_manager.get_B()
        except Exception as e:  # Catch all exceptions
            logger.error(f"Error during WarpX step: {e}")
            raise

    def restart(self, h5file):
        try:
            # ADIOS2 no-op; use HDF5 for particles
            self.warp.read_checkpoint(h5file)
            with h5py.File(h5file, 'r') as f:
                grp = f['particles']
                for name in self.species:
                    pos = grp[f"{name}_pos"][:]
                    vel = grp[f"{name}_vel"][:]
                    self.warp.clear_particles(name)
                    self.warp.add_particles(pos.tolist(), vel.tolist(), name)
        except Exception as e:
            logger.error(f"Error during restart: {e}")
            raise

    def get_total_current(self):
        try:
            Jz = self.field_manager.get_J()[2]
            return np.sum(Jz) * self.dx ** 2
        except Exception as e:
            logger.error(f"Error getting total current: {e}")
            return 0.0

    def get_field_slice(self, comp, axis='z', loc=0.0, downsample=(100, 100)):
        try:
            if comp == 'rho':
                arr = self.field_manager.get_rho()
            elif comp in ['Jx', 'Jy', 'Jz']:
                J = self.field_manager.get_J()
                arr = J[0] if comp == 'Jx' else J[1] if comp == 'Jy' else J[2]
            elif comp in ['Ex', 'Ey', 'Ez']:
                E = self.field_manager.get_E()
                arr = E[0] if comp == 'Ex' else E[1] if comp == 'Ey' else E[2]
            elif comp in ['Bx', 'By', 'Bz']:
                B = self.field_manager.get_B()
                arr = B[0] if comp == 'Bx' else B[1] if comp == 'By' else B[2]
            else:
                raise ValueError(f"Invalid field component: {comp}")

            if axis == 'z':
                idx = int((loc - self.domain_lower[2]) / self.dx)
                sl = arr[:, :, idx]
            else:
                sl = arr[self.grid_shape[0] // 2, :, :]
            sx, sy = downsample
            return sl[::max(1, sl.shape[0] // sx), ::max(1, sl.shape[1] // sy)]
        except Exception as e:
            logger.error(f"Error getting field slice: {e}")
            return np.zeros(downsample)

    def sample_particles(self, species, n=1000):
        try:
            pts = self.warp.get_particle_container(species)
            pos = np.array(pts.get_positions())
            vel = np.array(pts.get_velocities())
            idx = np.random.choice(len(pos), min(n, len(pos)), replace=False)
            return pos[idx], vel[idx]
        except Exception as e:
            logger.error(f"Error sampling particles: {e}")
            return np.array([]), np.array([])

    def stream_to_unity(self):
        try:
            data = {'time': self.warp.get_time(),
                    'current': self.get_total_current(),
                    'slice': self.get_field_slice('rho', 'z', self.unity_params.get('slice_plane', 0.0),
                                                  self.unity_params.get('downsample', (100, 100))).tolist(),
                    'particles': self.sample_particles(list(self.species.keys())[0],
                                                      self.unity_params.get('n_particles', 1000))[0].tolist()}
            self._queue.put(data)
        except Exception as e:
            logger.error(f"Error preparing data for Unity streaming: {e}")

    def _unity_stream_thread(self):
        try:
            conn = socket.create_connection((self.unity_params['host'], self.unity_params['port']))
            while True:
                msg = self._queue.get()
                if msg is None:
                    break
                conn.send(json.dumps(msg).encode())
            conn.close()
        except Exception as e:
            logger.error(f"Error in Unity streaming thread: {e}")

    def finalize(self):
        try:
            if hasattr(self, '_u_thread'):
                self._queue.put(None)
                self._u_thread.join()
            self.warp.finalize()
            self.writer.Close()
            logger.info("WarpXWrapper finalized.")
        except Exception as e:
            logger.error(f"Error during finalization: {e}")
