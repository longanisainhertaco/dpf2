import numpy as np
import sympy as sp
from mpi4py import MPI
import logging
import time

from config_schema import SimulationConfig
from module_registry import ModuleRegistry
from fluid_solver_high_order import FluidSolverHighOrder
from circuit import CircuitModel
from collision_model import CollisionModel
from radiation_model import RadiationModel
from pic_solver import PICSolver
from hybrid_controller import HybridController
from diagnostics import Diagnostics
from utils import FieldManager, SimulationState  # Import FieldManager and SimulationState

logger = logging.getLogger(__name__)

# Custom Exceptions
class ConfigurationError(Exception):
    pass

class InitializationError(Exception):
    pass

class RuntimeError(Exception):
    pass

def _generate_boundary_conditions(domain_lo, domain_hi, bc_config=None):
    """
    Generates boundary conditions for the fluid solver.

    Args:
        domain_lo (tuple): Lower corner of the domain.
        domain_hi (tuple): Upper corner of the domain.
        bc_config (dict, optional): Dictionary of boundary conditions. Defaults to None.

    Returns:
        dict: Dictionary of boundary conditions.
    """
    # Default boundary conditions
    default_bc = {
        'x_lo': 'symmetry',  'x_hi': 'conducting',
        'y_lo': 'symmetry',  'y_hi': 'conducting',
        'z_lo': 'fixed',     'z_hi': 'open'
    }

    # Override defaults with user-provided config
    if bc_config:
        default_bc.update(bc_config)

    return default_bc

class DPFSimulatorBackend:
    """
    Unified Dense Plasma Focus simulation orchestrator.
    """
    def __init__(self, config: dict):
        # Validate the config
        self.config = SimulationConfig(**config)

        # ——— Input validation ———
        dx_sym, t_sym = sp.symbols('dx sim_time', positive=True)
        if not (dx_sym.subs(dx_sym, self.config.dx) > 0 and t_sym.subs(t_sym, self.config.sim_time) > 0):
            raise ValueError(f"dx and sim_time must be positive: dx={self.config.dx}, sim_time={self.config.sim_time}")
        if (
            not isinstance(self.config.grid_shape, (tuple,list)) or
            len(self.config.grid_shape) != 3 or
            any((not isinstance(n,int) or n <= 0) for n in self.config.grid_shape)
        ):
            raise ValueError(f"grid_shape must be three positive ints: {self.config.grid_shape}")

        self.grid_shape = tuple(self.config.grid_shape)
        self.dx = self.config.dx
        self.sim_time = self.config.sim_time
        self.current_time = 0.0
        self.step_count = 0
        self.dt_min = 1e-12

        # dt_init can be a constant or a function(U,geom,dx) -> dt
        if self.config.dt_init:
            self.dt = self.config.dt_init
        else:
            self.dt = 0.0

        # Domain extents
        if self.config.geometry:
            if self.config.geometry.domain_hi is None:
                self.domain_hi = tuple(self.config.geometry.domain_lo[i] + self.grid_shape[i]*self.dx for i in range(3))
            else:
                self.domain_hi = self.config.geometry.domain_hi
            self.domain_lo = self.config.geometry.domain_lo
        else:
            self.domain_lo = (0.0,0.0,0.0)
            self.domain_hi = tuple(self.domain_lo[i] + self.grid_shape[i]*self.dx for i in range(3))

        # ——— Instantiate physics modules ———
        bc = _generate_boundary_conditions(self.domain_lo, self.domain_hi, self.config.geometry.boundary_conditions if self.config.geometry else None)

        # Create FieldManager
        self.field_manager = FieldManager(
            grid_shape=tuple(self.config.grid_shape),
            dx=self.config.dx,
            dy=self.config.dy,
            dz=self.config.dz,
            domain_lo=self.domain_lo,
            boundary_conditions=bc
        )

        self.fluid = FluidSolverHighOrder(
            geom=None,  # To be initialized later
            config=config,
            field_manager=self.field_manager
        )
        self.circuit   = CircuitModel(collision_model=None, **self.config.circuit.dict())
        self.collision = CollisionModel(**self.config.collision.dict()) if self.config.collision else None
        self.radiation = RadiationModel(geom=None, config=self.config.radiation.dict()) if self.config.radiation else None
        self.pic       = PICSolver(config=self.config.pic.dict(), field_manager=self.field_manager) if self.config.pic else None
        self.hybrid    = HybridController(
            config=self.config.hybrid.dict(),
            fluid_solver=self.fluid,
            pic_solver=self.pic,
            circuit_model=self.circuit,
            radiation_model=self.radiation,
            collision_model=self.collision,
            sheath_model=None, # TODO
            field_manager=self.field_manager
        ) if self.config.hybrid else None
        self.diagnostics = Diagnostics(
            hdf5_filename=self.config.diagnostics.hdf5_filename,
            config={**self.config.circuit.dict(), **self.config.collision.dict() if self.config.collision else {},
                    **self.config.radiation.dict() if self.config.radiation else {}, **self.config.pic.dict() if self.config.pic else {}, **self.config.hybrid.dict() if self.config.hybrid else {}},
            domain_lo=self.domain_lo,
            grid_shape=self.grid_shape,
            dx=self.dx,
            gamma=self.fluid.gamma
        ) if self.config.diagnostics else None

        # Telemetry & checkpoint intervals
        self.telemetry_callback = None
        self.checkpoint_interval = 1e-6
        self.next_checkpoint_time = self.checkpoint_interval
        self.full_output_interval = 10

        # MPI communicator for AMReX/WarpX
        self.comm = MPI.COMM_WORLD

    def run(self):
        """Advance the simulation through time until sim_time."""
        # Initialize WarpX / PIC context
        if self.pic:
            self.pic.initialize_warpx(comm=self.comm)

        # Create SimulationState
        self.state = SimulationState(
            grid_shape=tuple(self.config.grid_shape),
            dx=self.config.dx,
            dy=self.config.dy,
            dz=self.config.dz,
            domain_lo=self.domain_lo,
            boundary_conditions=self.config.geometry.boundary_conditions if self.config.geometry else {},
            field_manager=self.field_manager  # Pass FieldManager to SimulationState
        )

        # Main time-stepping loop
        while self.current_time < self.sim_time:
            self.step_count += 1

            try:
                # ——— Determine global dt ———
                dt = self.dt
                if not dt:
                    dt = self.compute_global_dt()

                # ——— Hybrid step (fluid/PIC orchestration) ———
                if self.hybrid:
                    self.hybrid.apply(self.state, dt)

                # ——— Circuit update with plasma L/R feedback ———
                try:
                    Lp, Rp = self.fluid.compute_plasma_LR()
                    self.circuit.step_trapezoidal(self.fluid, dt)
                except Exception as e:
                    logger.error(f"Error updating circuit: {e}")
                    raise

                # ——— Drive plasma with circuit current ———
                try:
                    I = self.circuit.get_current()
                    self.fluid.apply_circuit_drive(I)
                    if self.pic:
                        self.pic.apply_circuit_drive(I)
                except Exception as e:
                    logger.error(f"Error driving plasma with circuit current: {e}")
                    raise

                # ——— Collision & radiation sources ———
                try:
                    state    = self.fluid.get_state()
                    if self.collision:
                        coll_src = self.collision.compute_source_terms(state)
                        self.fluid.apply_collision_sources(coll_src)
                    if self.radiation:
                        rad_src  = self.radiation.compute_source_terms(state)
                        self.fluid.apply_radiation_sources(rad_src)
                except Exception as e:
                    logger.error(f"Error applying collision and radiation sources: {e}")
                    raise

                # ——— Advance fluid/PIC solvers ———
                try:
                    if self.hybrid:
                        self.hybrid.apply_sources_and_advance(dt, coll_src if self.collision else None, rad_src if self.radiation else None)
                    else:
                        self.fluid.step(dt)
                        if self.pic:
                            self.pic.step()
                except Exception as e:
                    logger.error(f"Error advancing fluid/PIC solvers: {e}")
                    raise

                # ——— AMReX regrid & dynamic load balancing ———
                try:
                    if self.fluid.should_regrid():
                        self.fluid.regrid()
                        self.fluid.do_load_balance()
                except Exception as e:
                    logger.error(f"Error during AMReX regridding: {e}")
                    raise

                # ——— Diagnostics recording ———
                if self.diagnostics:
                    if self.step_count % self.full_output_interval == 0:
                        self.diagnostics.record(self.current_time, self.circuit, self.fluid, self.pic, self.radiation)
                    else:
                        self.diagnostics.record(self.current_time, self.circuit, self.fluid, self.pic, self.radiation)

                # ——— Telemetry streaming ———
                if self.telemetry_callback:
                    try:
                        telemetry = self.diagnostics.to_json() if self.diagnostics else {}
                        self.telemetry_callback(telemetry)
                    except Exception as e:
                        logger.error(f"Error during telemetry streaming: {e}")

                # ——— Checkpoint / restart bookkeeping ———
                if self.current_time + dt >= self.next_checkpoint_time:
                    tag = f"chkpt_{self.current_time:.6f}"
                    self.fluid.write_checkpoint(tag)
                    if self.pic:
                        self.pic.write_checkpoint(tag)
                    if self.diagnostics:
                        self.diagnostics.link_checkpoint(tag)
                    self.next_checkpoint_time += self.checkpoint_interval

                # ——— Advance time ———
                self.current_time += dt

                # ——— Sanity check: no NaNs or negative ———
                s = self.fluid.get_state()
                assert np.all(s['density']>0), "Negative density detected"
                assert np.all(s['pressure']>0), "Negative pressure detected"

            except Exception as e:
                # Step back, reduce dt, retry or abort
                logger.warning(f"Exception occurred at t={self.current_time:.3e}: {e}. Reducing dt and retrying.")
                self.dt *= 0.5
                if self.dt < self.dt_min:
                    raise RuntimeError(f"Simulation aborted at t={self.current_time:.6e}: {e}")
                continue

        # Finalize WarpX / PIC
        if self.pic:
            self.pic.finalize_warpx()

        # Write out the full diagnostics HDF5
        if self.diagnostics:
            self.diagnostics.to_hdf5()

        # Print estimated total steps
        try:
            total_steps = int(np.ceil(self.sim_time / float(self.dt)))
            print(f"Estimated total steps: {total_steps}")
        except:
            pass

    def compute_global_dt(self):
        """Compute global dt satisfying CFL condition."""
        try:
            dt = self.fluid.cfl_dt(0.5)
            return dt
        except Exception as e:
            logger.error(f"Error computing CFL dt: {e}")
            return 0.0

    def get_state(self):
        state = {}
        state['provenance'] = self.config.provenance
        return state
