import numpy as np
import adios2
import resource
import logging
import time
from catalyst import CatalystPipeline
from dpf_simulator_full_backend import DPFSimulatorBackend
from fluid_solver_high_order import FluidSolver3DAMReX
from warpx_wrapper import WarpXInterface
from collision_model import CollisionModel
from radiation_model import RadiationModel
from circuit import CircuitModel
from diagnostics import Diagnostics, mu0
from sheath_model import PlasmaSheathFormation
from load_balance_metrics import LoadBalanceMetrics
from utils import FieldManager, SimulationState  # Import FieldManager and SimulationState

logger = logging.getLogger(__name__)

class DPFSimulatorAMReXBackend(DPFSimulatorBackend):
    def __init__(self, config: dict, field_manager: FieldManager): # Add field_manager
        super().__init__(config)
        io_cfg = config.get('io', {})

        # Telemetry variables (now configurable)
        self.telemetry_vars = config.get('telemetry', {}).get('variables', [
            'I', 'V', 'xray_power', 'neutron_rate',
            'dynamic_inductance', 'pinch_flag',
            'sheath_position', 'sheath_velocity',
            'peak_density', 'peak_temperature_e', 'peak_temperature_i',
            'total_plasma_energy', 'energy_residual',
            'divB_max', 'divB_l2',
            'cumulative_neutron_count', 'neutron_rate',
            'amr_levels', 'dt',
            'cell_balance_min', 'cell_balance_max',
            'particle_balance_min', 'particle_balance_max',
            'cpu_mem_usage_mb'
        ])

        # Plotfile intervals and variables
        self.coarse_interval = io_cfg.get('coarse_interval', self.checkpoint_interval)
        self.highres_interval = io_cfg.get('highres_interval', self.coarse_interval*5)
        self.coarse_plotfile_variables = io_cfg.get('coarse_plotfile_variables', ['rho', 'B'])
        self.fine_plotfile_variables = io_cfg.get('fine_plotfile_variables', ['rho', 'B', 'mom', 'energy'])
        self.next_coarse = self.coarse_interval
        self.next_highres = self.highres_interval

        # ADIOS2 for telemetry
        self.adios = adios2.ADIOS()
        self.bp_io = self.adios.DeclareIO("telemetryIO")
        self.bp_engine = self.bp_io.Open(io_cfg.get('telemetry_stream','telemetry.bp'),
                                         adios2.Mode.Write)

        # In-situ pipeline (check if it exists before instantiating)
        self.catalyst = CatalystPipeline(config.get('catalyst_script','catalyst.py'))

        # Create FieldManager
        self.field_manager = field_manager # Use the passed FieldManager

        # Instantiate solvers
        if self.mode in ('fluid','hybrid'):
            self.fluid_solver = FluidSolver3DAMReX(
                geom=None,  # To be initialized later
                config=config,
                field_manager=self.field_manager # Pass FieldManager to FluidSolver3DAMReX
            )
        if self.mode in ('pic','hybrid'):
            self.pic_solver = WarpXInterface(
                domain_lower=config['domain_lo'],
                domain_upper=config['domain_hi'],
                grid_shape=config['grid_shape'],
                dx=config['dx'],
                species_params=config['species'],
                pic_params=config['pic'],
                performance=config['performance'],
                fluid_callback=None,  # To be set later
                unity_params=config.get('unity', {}),
                circuit=self.circuit,
                field_manager=self.field_manager # Pass FieldManager to WarpXInterface
            )

        # Instantiate optional features
        self.sheath_model = None
        if config.get('enable_sheath_model', False):
            self.sheath_model = PlasmaSheathFormation(config.get('sheath_model', {}))

        self.load_balance_metrics = None
        if config.get('enable_load_balance_metrics', False):
            self.load_balance_metrics = LoadBalanceMetrics(self.fluid_solver)

        # Instantiate hybrid controller
        if self.mode=='hybrid':
            self.hybrid = HybridController(
                config=config['hybrid'],
                fluid_solver=self.fluid_solver,
                pic_solver=self.pic_solver,
                circuit_model=self.circuit,
                radiation_model=self.radiation,
                collision_model=self.collision,
                sheath_model=self.sheath_model,  # Ensure sheath model is passed
                field_manager=self.field_manager # Pass FieldManager to HybridController
            )

        # Instantiate diagnostics
        self.diagnostics = Diagnostics(
            hdf5_filename=config['diagnostics_params']['hdf5_filename'],
            config={**config['circuit_params'], **config['collision_params'] if config.get('collision_params') else {},
                    **config['radiation_params'] if config.get('radiation_params') else {}, **config['pic_params'] if config.get('pic_params') else {}, **config['hybrid_params'] if config.get('hybrid_params') else {}},
            domain_lo=self.domain_lo,
            grid_shape=self.grid_shape,
            dx=self.dx,
            gamma=self.fluid_solver.gamma,
            field_manager=self.field_manager # Pass FieldManager to Diagnostics
        )
        self.provenance = config.get('provenance', {})

    def run(self):
        self.t = getattr(self, 'time', 0.0)
        try:
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

            # Initialize solvers
            if self.mode in ('fluid','hybrid'):
                self.fluid_solver.geom = self.geom  # Initialize AMReX geometry
                self.fluid_solver.field_manager = self.field_manager  # Ensure FieldManager is set
            if self.mode in ('pic','hybrid'):
                self.pic_solver.fluid_callback = self.fluid_solver  # Set fluid callback
                self.pic_solver.state = self.state # Pass SimulationState to PIC solver
            if self.mode=='hybrid':
                self.hybrid.apply(self.state, self.dt)
                self._post_step(self.dt)
            else:
                while self.t < self.sim_time:
                    dt = self.compute_global_dt()
                    # Circuit half-step
                    if self.mode=='fluid':
                        Lp, Rp = self.fluid_solver.compute_plasma_LR()
                        self.circuit.half_step(self.state, dt)
                    else:
                        self.circuit.step_trapezoidal(self.state, dt)
                    # Physics
                    if self.mode=='fluid':
                        self.fluid_solver.step(dt)
                    else:
                        self.pic_solver.step()
                    # Circuit half-step complete
                    if self.mode=='fluid':
                        self.circuit.half_step(self.state, dt)
                    else:
                        self.circuit.step_trapezoidal(self.state, dt) # Assuming this is correct for PIC
                    self._post_step(dt)
                    self.t += dt
        except (ValueError, AttributeError, RuntimeError) as e:
            logger.error(f"[AMReX Backend] Aborted at t={self.t:.3e}: {e}")
            self._checkpoint(amrex_restart=True)
            raise
        finally:
            self._finalize()

    def _post_step(self, dt=None):
        try:
            # Record diagnostics
            self.diagnostics.record(self.t, self.circuit, self.fluid_solver, self.pic_solver, self.radiation)

            # Gather telemetry data
            tel = {}
            # basic signals
            tel['I'] = self.circuit.get_current()
            tel['V'] = self.circuit.get_voltage()
            # dynamic inductance & pinch flag
            Ldyn = self.diagnostics.get_latest()['mode_amplitudes'] if self.diagnostics.get_latest() and 'mode_amplitudes' in self.diagnostics.get_latest() else {}
            tel['dynamic_inductance'] = Ldyn
            tel['pinch_flag'] = int(self.diagnostics.get_latest()['dI_dt'] < 0) if self.diagnostics.get_latest() and 'dI_dt' in self.diagnostics.get_latest() else 0
            # sheath kinematics
            if self.sheath_model:
                self.sheath_model.apply(self.state, dt)
                tel['sheath_position'] = self.state.sheath_position
                tel['sheath_velocity'] = self.state.sheath_velocity
            # peak density & temperatures
            rho, Te, Ti = self.fluid_solver.get_peak_density_temperature()
            tel['peak_density'] = rho
            tel['peak_temperature_e'] = Te
            tel['peak_temperature_i'] = Ti
            # energy metrics
            tel['total_plasma_energy'] = self.fluid_solver.get_total_plasma_energy()
            tel['energy_residual'] = self.diagnostics.get_latest()['E_radiated'] if self.diagnostics.get_latest() and 'E_radiated' in self.diagnostics.get_latest() else 0.0
            # divergence errors
            divB_max, divB_l2 = self.diagnostics.compute_divergences()
            tel['divB_max'] = divB_max
            tel['divB_l2'] = divB_l2
            # neutron yield
            tel['cumulative_neutron_count'] = sum(self.diagnostics.get_latest()['histogram']) if self.diagnostics.get_latest() and 'histogram' in self.diagnostics.get_latest() else 0
            # AMR & dt
            tel['amr_levels'] = self.fluid_solver.num_levels()
            tel['dt'] = dt or self.dt_base
            # load balance metrics
            if self.load_balance_metrics:
                lb = self.load_balance_metrics.get_metrics()
                tel['cell_balance_min'] = lb['cell_min']
                tel['cell_balance_max'] = lb['cell_max']
                if hasattr(self, 'pic_solver'):
                    plb = self.pic_solver.get_particle_balance()
                    tel['particle_balance_min'] = plb['min']
                    tel['particle_balance_max'] = plb['max']
            # memory usage
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            tel['cpu_mem_usage_mb'] = mem

            # Stream telemetry
            self.bp_engine.BeginStep()
            for k in self.telemetry_vars:
                if k in tel:
                    self.bp_engine.Put(k, tel[k])
            self.bp_engine.EndStep()

            # In-situ Catalyst
            self.catalyst.execute(self.fluid_solver, getattr(self,'pic_solver',None))

            # Hierarchical outputs
            if self.t >= self.next_coarse:
                self.fluid_solver.write_plotfile(levels='coarse', metadata=self.provenance, variables=self.coarse_plotfile_variables)
                if self.mode!='fluid': self.pic_solver.write_plotfile(levels='coarse', metadata=self.provenance, variables=self.coarse_plotfile_variables)
                self.next_coarse += self.coarse_interval
            if self.t >= self.next_highres:
                self.fluid_solver.write_plotfile(levels='fine', metadata=self.provenance, variables=self.fine_plotfile_variables)
                if self.mode!='fluid':
                    self.pic_solver.write_plotfile(levels='fine', metadata=self.provenance, variables=self.fine_plotfile_variables)
                self.next_highres += self.highres_interval
        except Exception as e:
            logger.error(f"Error in _post_step: {e}")

    def _checkpoint(self, amrex_restart=False):
        if amrex_restart:
            self.fluid_solver.write_checkpoint(restart=True)
            if self.mode!='fluid': self.pic_solver.write_checkpoint(restart=True)
        else:
            super()._checkpoint()

    def _finalize(self):
        self.bp_engine.Close()
        self.diagnostics.to_hdf5()
        logger.info(f"[AMReX Backend] Simulation complete at t={self.t:.3e}")

    def compute_global_dt(self):
        return super().compute_global_dt()

    def get_state(self):
        state = super().get_state()
        state['provenance'] = self.provenance
        return state
