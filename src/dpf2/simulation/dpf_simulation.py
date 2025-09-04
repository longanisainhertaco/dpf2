#!/usr/bin/env python3
"""
DPF Simulation Launcher: Comprehensive multi-physics configuration.
"""
import os
import sys
import time
import json
import socket
import subprocess
import logging
import argparse
import numpy as np
import random
from datetime import datetime
from typing import Dict

try:  # Prefer package-local imports
    from .config_schema import SimulationConfig, FieldManagerConfig, PICConfig, AMRConfig
except Exception:  # pragma: no cover - fallback when loaded as a script
    from config_schema import SimulationConfig, FieldManagerConfig  # type: ignore
    class PICConfig:  # type: ignore
        pass
    class AMRConfig:  # type: ignore
        enable: bool = False
        max_level: int = 1
        refinement_threshold: float = 1.0
        diag_interval: int = 10

_AMR = AMRConfig
from .module_registry import ModuleRegistry
from .collision_model import CollisionModel
from .radiation_model import RadiationModel
from .hybrid_controller import HybridController
from .eos_selector import select_eos
from .solver_selector import select_solver
from .circuit import CircuitModel
from .utils import FieldManager, SimulationState
from ..diagnostics import Diagnostics
from .pic_solver import PICSolver
from ..core.bases import CouplingState
from ..materials import (
    MaterialLibrary,
    ComponentMaterialState,
    MaterialDamageModel,
)
try:
    from ..exceptions import SimulationRuntimeError
except Exception:  # pragma: no cover - fallback for standalone usage
    class SimulationRuntimeError(RuntimeError):
        pass

logger = logging.getLogger("DPFSimulationWrapper")

# Custom Exceptions
class ConfigurationError(Exception):
    pass

class InitializationError(Exception):
    pass


class DPFSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.modules = {}
        self.step_count = 0
        self.current_time = 0.0
        self.dt = self.config.dt_init
        self.amr_config = getattr(self.config, "amr", _AMR())
        self.material_model: MaterialDamageModel | None = None

        # Initialize modules
        self.registry = ModuleRegistry()
        self.register_modules()
        self.initialize_modules()

    def register_modules(self):
        """Registers available modules with the registry."""
        self.registry.register(CollisionModel, field_manager_required=True)
        self.registry.register(RadiationModel, field_manager_required=True)
        self.registry.register(HybridController, field_manager_required=True)

    def initialize_modules(self):
        """Initializes modules based on the configuration."""
        try:
            # Instantiate EOS & solver
            self.eos = select_eos(backend=self.config.eos_backend,
                                     table_file=self.config.table_file,
                                     mixture_fractions=(self.config.mixture_fractions if self.config.enable_eos_mixture else None))
            # Create FieldManager
            self.field_manager = FieldManager(
                grid_shape=tuple(self.config.grid_shape),
                dx=self.config.dx,
                dy=self.config.dy,
                dz=self.config.dz,
                domain_lo=self.config.domain_lo,
                boundary_conditions=self.config.field_manager.boundary_conditions
            )

            # Create SimulationState
            self.state = SimulationState(
                grid_shape=tuple(self.config.grid_shape),
                dx=self.config.dx,
                dy=self.config.dy,
                dz=self.config.dz,
                domain_lo=self.config.domain_lo,
                boundary_conditions=self.config.field_manager.boundary_conditions,
                field_manager=self.field_manager,
                amr_config=self.amr_config,
            )

            self.solver = select_solver(
                backend=self.config.solver_backend,
                config={
                    "grid_shape": tuple(self.config.grid_shape),
                    "dx": self.config.dx,
                    "dy": self.config.dy,
                    "dz": self.config.dz,
                },
                field_manager=self.field_manager,
            )

            self.pic_solver = None
            if self.config.pic:
                pic_cfg = self.config.pic.dict()
                pic_cfg.update(
                    {
                        "amr": self.amr_config.enable,
                        "density_threshold": self.amr_config.refinement_threshold,
                    }
                )
                self.pic_solver = PICSolver(
                    config=PICConfig(**pic_cfg), field_manager=self.field_manager
                )

            if self.config.collision:
                self.modules["collision"] = self.registry.create(
                    CollisionModel, self.config.collision.dict(), field_manager=self.field_manager
                )
            if self.config.radiation:
                self.modules["radiation"] = self.registry.create(
                    RadiationModel, self.config.radiation.dict(), field_manager=self.field_manager
                )

            self.circuit = CircuitModel(
                collision_model=self.modules.get("collision"),
                field_manager=self.field_manager,
                **self.config.circuit.dict(),
            )

            if self.config.hybrid:
                hybrid_config = self.config.hybrid.dict()
                hybrid_config["fluid_solver"] = self.solver
                hybrid_config["pic_solver"] = self.pic_solver
                hybrid_config["circuit_model"] = self.circuit
                hybrid_config["radiation_model"] = self.modules.get("radiation")
                hybrid_config["field_manager"] = self.field_manager
                self.modules["hybrid"] = self.registry.create(
                    HybridController, hybrid_config, field_manager=self.field_manager
                )
            if self.config.diagnostics:
                self.modules["diagnostics"] = Diagnostics(
                    hdf5_filename=self.config.diagnostics.hdf5_filename,
                    config={
                        **self.config.circuit.dict(),
                        **(self.config.collision.dict() if self.config.collision else {}),
                        **(self.config.radiation.dict() if self.config.radiation else {}),
                        **(self.config.pic.dict() if self.config.pic else {}),
                        **(self.config.hybrid.dict() if self.config.hybrid else {}),
                    },
                    domain_lo=self.config.domain_lo,
                    grid_shape=self.config.grid_shape,
                    dx=self.config.dx,
                    gamma=self.solver.gamma,
                    field_manager=self.field_manager,
                )

            materials_cfg = getattr(self.config, "materials", None)
            if materials_cfg and materials_cfg.components:
                component_states: Dict[str, ComponentMaterialState] = {}
                for comp, mat_ref in materials_cfg.components.items():
                    mat = MaterialLibrary.get(mat_ref.material_id)
                    init = materials_cfg.initial_state.get(comp, {})
                    component_states[comp] = ComponentMaterialState(
                        material=mat,
                        erosion=float(init.get("erosion", 0.0)),
                        film_thickness=float(init.get("film_thickness", 0.0)),
                    )
                self.material_model = MaterialDamageModel(
                    component_states, plasma_model=self.modules.get("collision")
                )

        except Exception as e:
            raise InitializationError(f"Failed to initialize modules: {e}")

    def run(self):
        """Runs the simulation."""
        try:
            while self.current_time < self.config.sim_time:
                # --- determine timestep ---
                if hasattr(self.solver, "compute_optimal_dt"):
                    self.dt = self.solver.compute_optimal_dt()
                elif hasattr(self.solver, "compute_dt"):
                    self.dt = self.solver.compute_dt()
                if self.dt is None or self.dt <= 0.0:
                    raise RuntimeError("Invalid time step")
                # clip to remaining simulation time
                if self.current_time + self.dt > self.config.sim_time:
                    self.dt = self.config.sim_time - self.current_time

                # --- advance primary solver or hybrid controller ---
                if "hybrid" in self.modules:
                    try:
                        self.modules["hybrid"].apply(self.state, self.dt)
                    except Exception as exc:
                        logger.error(f"Hybrid controller error: {exc}")
                else:
                    self.solver.step(self.dt)
                    if self.pic_solver:
                        self.pic_solver.step()

                # --- material damage model ---
                if self.material_model:
                    self.material_model.apply(self.solver, self.dt)

                # --- collision and radiation modules ---
                for name in ("collision", "radiation"):
                    module = self.modules.get(name)
                    if module:
                        try:
                            module.apply(self.state, self.dt)
                        except Exception as exc:
                            logger.error(f"{name.capitalize()} module error: {exc}")

                # --- circuit update ---
                try:
                    # Obtain the plasma current and any induced back‑EMF from the
                    # plasma solver if available.  Fallback to the field manager
                    # for the current and zero EMF when the solver does not
                    # expose them.  Likewise retrieve optional plasma feedback
                    # terms such as a time varying inductance.
                    current = getattr(self.solver, "current", None)
                    if callable(current):
                        current = current()
                    if current is None:
                        try:
                            current = self.field_manager.get_J()
                        except Exception:
                            current = 0.0

                    back_emf = getattr(self.solver, "back_emf", None)
                    if callable(back_emf):
                        back_emf = back_emf()
                    if back_emf is None:
                        back_emf = 0.0

                    feedback = getattr(self.solver, "circuit_feedback", None)

                    if isinstance(feedback, CouplingState):
                        coupling = feedback
                        coupling.current = current
                        coupling.voltage = (
                            self.circuit.get_voltage()
                            if hasattr(self.circuit, "get_voltage")
                            else 0.0
                        )
                    else:
                        feedback = feedback or {}
                        coupling = CouplingState(
                            Lp=feedback.get("Lp", 0.0),
                            emf=feedback.get("emf", 0.0),
                            current=current,
                            voltage=(
                                self.circuit.get_voltage()
                                if hasattr(self.circuit, "get_voltage")
                                else 0.0
                            ),
                        )

                    self.circuit.step(coupling, back_emf, self.dt)
                except Exception as exc:
                    logger.error(f"Circuit step failed: {exc}")

                # --- diagnostics and checkpointing ---
                checkpoint_data = {}
                for name, module in self.modules.items():
                    if hasattr(module, "checkpoint"):
                        try:
                            checkpoint_data[name] = module.checkpoint()
                        except Exception as exc:
                            logger.error(f"Checkpoint failed for {name}: {exc}")
                diagnostics = self.modules.get("diagnostics")
                if diagnostics:
                    try:
                        diagnostics.record(self.current_time, self.circuit, self.solver, self.pic_solver, self.modules.get("radiation"), checkpoint_id=self.step_count)
                        if hasattr(diagnostics, "checkpoints"):
                            diagnostics.checkpoints.append(checkpoint_data)
                    except Exception as exc:
                        logger.error(f"Diagnostics error: {exc}")

                # --- advance time ---
                self.step_count += 1
                self.current_time += self.dt
                logger.info(f"Step {self.step_count}: time={self.current_time:.3e}")

        except Exception as e:
            raise SimulationRuntimeError(
                f"Simulation failed at step {self.step_count}: {e}"
            )

    def finalize(self):
        """Finalizes the simulation."""
        logger.info("Simulation completed.")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="DPF Simulation Launcher")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Global logging level")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Enable OpenCensus tracing for simulation stages",
    )
    args = parser.parse_args()
    return args

def load_config_from_json(filepath):
    """Loads configuration from a JSON file and validates it."""
    try:
        with open(filepath, 'r') as f:
            config_data = json.load(f)
            config = SimulationConfig(**config_data)  # Validate the config
            return config
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Error decoding JSON from {filepath}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error validating configuration: {e}")

def main():
    args = parse_arguments()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Seed RNGs
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Load and validate configuration
    try:
        config = load_config_from_json(args.config_file)
    except ConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    tracer = None
    if args.enable_tracing:
        try:
            from opencensus.trace.tracer import Tracer

            tracer = Tracer()
        except ModuleNotFoundError:
            logger.error(
                "OpenCensus is required for tracing but is not installed."
            )
            sys.exit(1)

    # Instantiate and run the simulation
    try:
        if tracer:
            with tracer.span(name="initialize"):
                sim = DPFSimulation(config)
            with tracer.span(name="run"):
                sim.run()
            with tracer.span(name="finalize"):
                sim.finalize()
        else:
            sim = DPFSimulation(config)
            sim.run()
            sim.finalize()
    except InitializationError as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    except SimulationRuntimeError as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
