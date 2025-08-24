import os
import sys
import json
import time
import logging
from typing import List, Optional
import numpy as np
import pywarpx
from pywarpx import picmi

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("AdvancedWarpXSimulation")

class AdvancedWarpXSimulation:
    """
    Advanced production simulation using WarpX for plasma research.

    This simulation leverages the PICMI interface to configure and run an
    adaptive mesh electromagnetic simulation with adaptive mesh refinement.
    It sets up the simulation domain, plasma species, diagnostics, and benchmarking.
    """

    def __init__(self, config_path: str, field_manager=None):
        """
        Initialize simulation from a JSON configuration file.

        Args:
            config_path (str): Path to JSON file containing simulation parameters.
        """
        self.config_path = config_path
        self._load_config()
        self._validate_config()
        self.field_manager = field_manager

        os.environ["WARPX_AMR_MAX_LEVEL"] = str(self.config["amr_levels"])
        os.environ["WARPX_REFINE_THRESH"] = str(self.config["refinement_threshold"])

        # Initialize simulation domain
        grid_length = [self.config["prob_hi"][i] - self.config["prob_lo"][i] for i in range(3)]
        self.sim = picmi.ElectromagneticSimulation(
            solver_type="FiniteDifferenceSolver",
            grid_length=grid_length,
            grid_npoints=self.config["ncell"],
            dt=self.config["dt"],
            max_steps=self.config["max_steps"],
        )

        self._setup_species()
        self._setup_diagnostics()

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        logger.info(f"Configuration loaded from {self.config_path}")

    def _validate_config(self):
        required_keys = ["ncell", "prob_lo", "prob_hi", "dt", "max_steps",
                         "amr_levels", "refinement_threshold", "diag_interval"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration parameter: {key}")

    def _setup_species(self):
        """Configure plasma species."""
        bounds = self.config["prob_lo"] + self.config["prob_hi"]
        distribution = picmi.UniformDistribution(
            density=1e19,
            xlo=bounds[0], xhi=bounds[3],
            ylo=bounds[1], yhi=bounds[4],
            zlo=bounds[2], zhi=bounds[5],
        )

        self.electron = picmi.Species(
            particle_type="electron",
            name="electron",
            charge=-1.60217662e-19,
            mass=9.10938356e-31,
            initial_distribution=distribution
        )

        self.ion = picmi.Species(
            particle_type="ion",
            name="ion",
            charge=1.60217662e-19,
            mass=1.6726219e-27,
            initial_distribution=distribution
        )

        self.sim.add_species(self.electron)
        self.sim.add_species(self.ion)

    def _setup_diagnostics(self):
        """Setup diagnostics for fields and particles."""
        os.makedirs("diags/fields", exist_ok=True)
        os.makedirs("diags/particles", exist_ok=True)

        self.field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            interval=self.config["diag_interval"],
            data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
            write_dir="diags/fields",
        )
        self.sim.add_diagnostic(self.field_diag)

        self.electron_diag = picmi.ParticleDiagnostic(
            name="electron_diag",
            species=self.electron,
            interval=self.config["diag_interval"],
            write_dir="diags/particles",
        )
        self.sim.add_diagnostic(self.electron_diag)

    def initialize(self):
        """Initialize the simulation."""
        try:
            logger.info("Initializing simulation...")
            self.sim.initialize_inputs()
            self.sim.initialize_simulation()
            logger.info("Simulation initialized successfully.")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def run(self, state: SimulationState):
        """Run the simulation loop."""
        try:
            logger.info("Starting simulation...")
            start_time = time.perf_counter()
            for step in range(self.config["max_steps"]):
                pywarpx.warpx.step()
                if step % self.config["diag_interval"] == 0:
                    logger.info(f"Step {step} completed.")
            end_time = time.perf_counter()
            logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

    def visualize_diagnostics(self):
        """Placeholder for diagnostics visualization (extend as needed)."""
        logger.info("Visualizing diagnostics is not yet implemented.")

    def refine_grid(self, state: SimulationState):
        """Refines the grid based on particle density."""
        try:
            if not self.config["enable_amr"]:
                return

            rho = state.density  # Access density from SimulationState

            # Implement AMR logic based on density
            # Example: Refine where density exceeds a threshold
            for level in range(self.config["amr_levels"]):
                for i in range(self.config["ncell"][0]):
                    for j in range(self.config["ncell"][1]):
                        for k in range(self.config["ncell"][2]):
                            if rho[i, j, k] > self.config["refinement_threshold"]:
                                # Refine the grid at this location
                                # (Implementation depends on your AMR library)
                                logger.debug(f"Refining grid at ({i}, {j}, {k})")
        except Exception as e:
            logger.error(f"Error during grid refinement: {e}")

# Usage: run with a config file (example_config.json)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python simulation.py <config_path.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    sim = AdvancedWarpXSimulation(config_file)
    sim.initialize()
    # Create a dummy SimulationState
    class DummySimulationState:
        def __init__(self, grid_shape):
            self.density = np.zeros(grid_shape)
    state = DummySimulationState(sim.config["ncell"])
    sim.run(state)
    sim.visualize_diagnostics()
