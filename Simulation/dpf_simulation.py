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
import opencensus.trace as trace
from datetime import datetime

from config_schema import SimulationConfig, FieldManagerConfig  # Import FieldManagerConfig
from module_registry import ModuleRegistry
from collision_model import CollisionModel
from radiation_model import RadiationModel
from hybrid_controller import HybridController
from eos_selector import select_eos
from solver_selector import select_solver
from circuit import CircuitModel
from utils import FieldManager, SimulationState # Import FieldManager and SimulationState
from diagnostics import Diagnostics

logger = logging.getLogger("DPFSimulationWrapper")

# Custom Exceptions
class ConfigurationError(Exception):
    pass

class InitializationError(Exception):
    pass

class RuntimeError(Exception):
    pass

class DPFSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.modules = {}
        self.step_count = 0
        self.current_time = 0.0
        self.dt = self.config.dt_init

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
                field_manager=self.field_manager  # Pass FieldManager to SimulationState
            )

            self.solver = select_solver(backend=self.config.solver_backend,
                                       grid_shape=tuple(self.config.grid_shape),
                                       dx=self.config.dx,
                                       nthreads=self.config.nthreads,
                                       gpu_block_size=self.config.gpu_block_size,
                                       implicit_tol=self.config.implicit_tol,
                                       max_iter=self.config.max_implicit_iter,
                                       field_manager=self.field_manager # Pass FieldManager to solver
                                       )

            # Instantiate modules
            if self.config.collision:
                self.modules['collision'] = self.registry.create(CollisionModel, self.config.collision.dict(), field_manager=self.field_manager)
            if self.config.radiation:
                self.modules['radiation'] = self.registry.create(RadiationModel, self.config.radiation.dict(), field_manager=self.field_manager)
            if self.config.hybrid:
                hybrid_config = self.config.hybrid.dict()
                hybrid_config['fluid_solver'] = self.solver
                hybrid_config['pic_solver'] = None # TODO: add pic solver
                hybrid_config['circuit_model'] = self.circuit
                hybrid_config['radiation_model'] = self.modules.get('radiation')
                hybrid_config['field_manager'] = self.field_manager # Pass FieldManager to HybridController
                self.modules['hybrid'] = self.registry.create(HybridController, hybrid_config, field_manager=self.field_manager)
            if self.config.diagnostics:
                self.modules['diagnostics'] = Diagnostics(
                    hdf5_filename=self.config.diagnostics.hdf5_filename,
                    config={**self.config.circuit.dict(), **self.config.collision.dict() if self.config.collision else {},
                            **self.config.radiation.dict() if self.config.radiation else {}, **self.config.pic.dict() if self.config.pic else {}, **self.config.hybrid.dict() if self.config.hybrid else {}},
                    domain_lo=self.config.domain_lo,
                    grid_shape=self.config.grid_shape,
                    dx=self.config.dx,
                    gamma=self.solver.gamma,
                    field_manager=self.field_manager
                )

            # Instantiate circuit
            self.circuit = CircuitModel(collision_model=self.modules.get('collision'), field_manager=self.field_manager, **self.config.circuit.dict())

        except Exception as e:
            raise InitializationError(f"Failed to initialize modules: {e}")

    def run(self):
        """Runs the simulation."""
        try:
            # Main loop
            while self.current_time < self.config.sim_time:
                self.step_count += 1
                self.solver.step(self.dt)
                self.current_time += self.dt
                logger.info(f"Step {self.step_count}: time={self.current_time:.3e}")

        except Exception as e:
            raise RuntimeError(f"Simulation failed at step {self.step_count}: {e}")

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

    # Instantiate and run the simulation
    try:
        sim = DPFSimulation(config)
        sim.run()
        sim.finalize()
    except InitializationError as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
