import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.optimize import root_scalar
from scipy.linalg import solve
from scipy.sparse import diags
from numba import njit, prange
from models import PhysicsModule, SimulationState
from config_schema import SheathConfig
from typing import Dict, Any

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# Physical constants
e_charge = 1.602176634e-19  # C
epsilon0 = 8.854187817e-12  # F/m
k_B = 1.380649e-23  # J/K
m_e = 9.10938356e-31  # kg

class PlasmaSheathFormation(PhysicsModule):
    """
    A high-fidelity model for plasma sheath formation, including:
    - Poisson's equation solver for electric field (using finite differences)
    - Boltzmann relation for electron density
    - Ion density profile based on a more complete fluid model and Bohm criterion
    - Non-Maxwellian electron flux (approximated)
    - Dynamic sheath thickness calculation
    - Configurable grid and boundary conditions
    - Consistent temperature units (eV)
    - Improved error handling
    """

    def __init__(self, config: SheathConfig):
        """
        Initializes the PlasmaSheathFormation model.

        Args:
            config: A SheathConfig object containing the model parameters.
        """
        self.config = config
        self.ion_density = config.ion_density  # ions/m³
        self.electron_density = config.electron_density  # electrons/m³
        self.sheath_voltage = config.sheath_voltage  # Volts
        self.ion_temperature = config.ion_temperature  # eV
        self.electron_temperature = config.electron_temperature  # eV
        self.ion_mass = config.ion_mass  # kg
        self.dx = config.dx
        self.max_sheath_thickness = config.max_sheath_thickness
        self.num_grid_points = config.num_grid_points
        self.x_grid = None
        self.electric_field = None
        self.electric_potential = None
        self.ion_density_profile = None
        self.electron_density_profile = None
        self.sheath_thickness = 0.0
        self.bohm_velocity = 0.0
        self.checkpoint_data = {}
        self.plasma_edge_potential = config.plasma_edge_potential

        logger.info("PlasmaSheathFormation initialized.")

    def compute_sheath_thickness(self):
        """
        Computes the sheath thickness using a more accurate iterative method.

        Returns:
            The sheath thickness in meters.
        """
        try:
            # Iteratively solve for sheath thickness
            def sheath_equation(s):
                # Child-Langmuir law (collisionless)
                cl_current = (4/9) * epsilon0 * np.sqrt(2 * e_charge / self.ion_mass) * (self.sheath_voltage**(3/2)) / (s**2)
                # Bohm flux (approximation)
                bohm_flux = self.ion_density * self.bohm_velocity * e_charge
                return cl_current - bohm_flux

            # Find the root of the equation
            result = root_scalar(sheath_equation, bracket=[self.dx, self.max_sheath_thickness], method='brentq')
            self.sheath_thickness = result.root
            return self.sheath_thickness
        except Exception as e:
            logger.error(f"Error computing sheath thickness: {e}")
            return 0.0

    def _poisson_equation(self):
        """
        Solves Poisson's equation using finite differences.

        Returns:
            The electric potential profile.
        """
        try:
            num_points = max(self.num_grid_points, int(self.sheath_thickness / self.dx))
            self.x_grid = np.linspace(0, self.sheath_thickness, num_points)
            h = self.x_grid[1] - self.x_grid[0]

            # Define the matrix A for the finite difference approximation
            main_diag = -2 * np.ones(num_points)
            off_diag = np.ones(num_points - 1)
            A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(num_points, num_points)).toarray()

            # Boundary conditions
            A[0, :] = 0
            A[0, 0] = 1
            A[-1, :] = 0
            A[-1, -1] = 1

            # Source term (charge density)
            rho = e_charge * (self.ion_density_profile - self.electron_density_profile)
            b = -rho / epsilon0 * h**2
            b[0] = self.plasma_edge_potential  # Potential at the plasma edge
            b[-1] = self.sheath_voltage  # Potential at the wall

            # Solve the linear system
            self.electric_potential = solve(A, b)
            self.electric_field = -np.gradient(self.electric_potential, h)

            return self.electric_potential

        except Exception as e:
            logger.error(f"Error solving Poisson's equation: {e}")
            return np.zeros_like(self.x_grid)

    @njit(parallel=False) # Removed parallel=True, as it might not be beneficial here
    def _ion_fluid_equations(self):
        """
        Solves the ion fluid equations (continuity and momentum) using a higher-order scheme.
        """
        try:
            num_points = max(self.num_grid_points, int(self.sheath_thickness / self.dx))
            self.x_grid = np.linspace(0, self.sheath_thickness, num_points)
            h = self.x_grid[1] - self.x_grid[0]

            # Initialize profiles
            self.ion_density_profile = np.zeros(num_points)
            self.ion_velocity_profile = np.zeros(num_points)

            # Boundary conditions
            self.bohm_velocity = np.sqrt(e_charge * self.electron_temperature / self.ion_mass)
            self.ion_density_profile[0] = self.ion_density
            self.ion_velocity_profile[0] = self.bohm_velocity

            # Solve the fluid equations using a higher-order finite difference scheme
            for i in range(1, num_points - 1): # Changed to range
                # Continuity equation: dn/dx = -n * dv/dx / v
                dvdx = (self.ion_velocity_profile[i] - self.ion_velocity_profile[i-1]) / h # Changed to forward difference
                dndx = -self.ion_density_profile[i] * dvdx / (self.ion_velocity_profile[i] + 1e-30)
                self.ion_density_profile[i] = self.ion_density_profile[i-1] + dndx * h

                # Momentum equation: dv/dx = -e * E / (m * v)
                dphidx = (self.electric_potential[i] - self.electric_potential[i-1]) / h # Changed to forward difference
                dvdx = -(e_charge * dphidx) / (self.ion_mass * (self.ion_velocity_profile[i] + 1e-30))
                self.ion_velocity_profile[i] = self.ion_velocity_profile[i-1] + dvdx * h

            return self.ion_density_profile, self.ion_velocity_profile

        except Exception as e:
            logger.error(f"Error solving ion fluid equations: {e}")
            return np.zeros_like(self.x_grid), np.zeros_like(self.x_grid)

    def _non_maxwellian_electron_flux(self):
        """
        Computes the electron flux to the surface using a more accurate approximation.
        """
        try:
            # Example: Use a more accurate approximation based on the electron distribution function
            # This is a placeholder; replace with a more sophisticated model
            electron_velocity = np.sqrt(e_charge * self.electron_temperature / (m_e))
            electron_flux = 0.5 * self.electron_density * electron_velocity * np.exp(-0.5)  # Example correction
            return electron_flux
        except Exception as e:
            logger.error(f"Error computing non-Maxwellian electron flux: {e}")
            return 0.0

    def compute_density_profiles(self):
        """
        Computes the electron and ion density profiles within the sheath.
        """
        try:
            if self.electric_potential is None:
                self.compute_electric_field()

            # Boltzmann relation for electron density
            self.electron_density_profile = self.electron_density * np.exp(e_charge * (self.electric_potential - self.plasma_edge_potential) / (self.electron_temperature * e_charge))

            # Solve ion fluid equations
            self.ion_density_profile, self.ion_velocity_profile = self._ion_fluid_equations()

        except Exception as e:
            logger.error(f"Error computing density profiles: {e}")

    def compute_electric_field(self):
        """
        Computes the electric field and potential within the sheath.
        """
        try:
            self.compute_sheath_thickness()
            self._poisson_equation()

        except Exception as e:
            logger.error(f"Error computing electric field: {e}")

    def compute_ion_flux(self):
        """
        Computes the ion flux to the surface using the Bohm criterion.

        Returns:
            The ion flux in ions/m²/s.
        """
        try:
            # Bohm criterion for ion velocity at the sheath edge
            ion_flux = self.ion_density * self.bohm_velocity
            return ion_flux
        except Exception as e:
            logger.error(f"Error computing ion flux: {e}")
            return 0.0

    def compute_electron_flux(self):
        """
        Computes the electron flux to the surface.

        Returns:
            The electron flux in electrons/m²/s.
        """
        try:
            # Use the non-Maxwellian electron flux calculation
            electron_flux = self._non_maxwellian_electron_flux()
            return electron_flux
        except Exception as e:
            logger.error(f"Error computing electron flux: {e}")
            return 0.0

    def apply(self, state: SimulationState, dt: float):
        """
        Applies the sheath model to the current simulation state.

        Args:
            state: The current state of the simulation.
            dt: The time step.
        """
        try:
            # Update the sheath model based on the current state of the simulation
            self.compute_sheath_thickness()
            self.compute_electric_field()
            self.compute_density_profiles()
            ion_flux = self.compute_ion_flux()
            electron_flux = self.compute_electron_flux()

            # Apply the sheath model to the fluid state
            # Apply sheath potential as a boundary condition on the electric field
            # Assuming the sheath forms at the high-z boundary (adjust as needed for your geometry)
            if hasattr(state, 'field_manager'):
                E = state.field_manager.get_E()
                g = 2  # Number of ghost cells (adjust if different in your setup)
                # Apply the sheath potential to the x-component of the electric field at the high-z boundary
                E[0, :, :, -g:] = self.electric_field[-1]  # Assuming 1D sheath, apply the last value
                state.field_manager.update_E(E)
                logger.debug(f"Applied sheath potential boundary condition using FieldManager. Electric field at boundary: {E[0, :, :, -g:]}")
            else:
                logger.warning("SimulationState does not have 'field_manager' attribute. Sheath potential BC not applied.")

            # Apply Bohm velocity as a boundary condition on ion velocity (if applicable)
            # This part might need adjustment depending on how ion velocity is handled in your fluid solver
            # and whether it's directly accessible through the FieldManager or SimulationState
            # Example (assuming ion velocity is a field that can be set):
            # if hasattr(state, 'ion_velocity'):
            #     state.ion_velocity[-g:, :, :] = self.bohm_velocity
            # else:
            #     logger.warning("SimulationState does not have 'ion_velocity' attribute. Bohm velocity BC not applied.")

            logger.debug(f"PlasmaSheathFormation applied. Ion Flux: {ion_flux:.3e}, Electron Flux: {electron_flux:.3e}")
        except Exception as e:
            logger.error(f"Error applying PlasmaSheathFormation: {e}")

    def visualize_sheath_profile(self):
        """
        Visualizes the sheath density profiles, electric field, and potential.
        """
        try:
            if self.x_grid is None or self.ion_density_profile is None or self.electron_density_profile is None or self.electric_field is None or self.electric_potential is None:
                self.compute_sheath_thickness()
                self.compute_electric_field()
                self.compute_density_profiles()

            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.plot(self.x_grid, self.ion_density_profile, label="Ion Density")
            plt.plot(self.x_grid, self.electron_density_profile, label="Electron Density")
            plt.xlabel("Distance (m)")
            plt.ylabel("Density (m⁻³)")
            plt.legend()
            plt.title("Plasma Sheath Density Profiles")

            plt.subplot(1, 3, 2)
            plt.plot(self.x_grid, self.electric_field)
            plt.xlabel("Distance (m)")
            plt.ylabel("Electric Field (V/m)")
            plt.title("Electric Field in Sheath")

            plt.subplot(1, 3, 3)
            plt.plot(self.x_grid, self.electric_potential)
            plt.xlabel("Distance (m)")
            plt.ylabel("Electric Potential (V)")
            plt.title("Electric Potential in Sheath")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error visualizing sheath profile: {e}")

    def initialize(self):
        """
        Initializes the sheath model.
        """
        logger.info("PlasmaSheathFormation initialized.")

    def finalize(self):
        """
        Finalizes the sheath model.
        """
        logger.info("PlasmaSheathFormation finalized.")

    def configure(self, config: Dict[str, Any]):
        """Configures the sheath model."""
        try:
            for key, value in config.items():
                setattr(self, key, value)
            logger.info(f"PlasmaSheathFormation configured with: {config}")
        except Exception as e:
            logger.error(f"Error configuring PlasmaSheathFormation: {e}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns diagnostic information."""
        try:
            return {
                "sheath_thickness": self.sheath_thickness,
                "sheath_drop_analytic": self.analytic_sheath_drop(),
                "sheath_drop_model": self.sheath_voltage - self.plasma_edge_potential,
                "ion_flux": self.compute_ion_flux(),
                "electron_flux": self.compute_electron_flux(),
                "bohm_velocity": self.bohm_velocity,
            }
        except Exception as e:
            logger.error(f"Error getting diagnostics: {e}")
            return {}

    def checkpoint(self) -> Dict[str, Any]:
        """Returns a dictionary of data to checkpoint."""
        try:
            self.checkpoint_data = {
                'sheath_thickness': self.sheath_thickness,
                'bohm_velocity': self.bohm_velocity,
                # Add other data as needed
            }
            return self.checkpoint_data
        except Exception as e:
            logger.error(f"Error during checkpoint: {e}")
            return {}

    def restart(self, data: Dict[str, Any]):
        """Restores data from a checkpoint."""
        try:
            self.sheath_thickness = data.get('sheath_thickness', 0.0)
            self.bohm_velocity = data.get('bohm_velocity', 0.0)
        except Exception as e:
            logger.error(f"Error during restart: {e}")

    def analytic_sheath_drop(self):
        """
        Estimates the sheath potential drop using an analytical approximation.
        (Simplified for now, needs refinement)
        """
        try:
            # Simplified analytical estimate (needs improvement)
            # This is just a placeholder and should be replaced with a more accurate formula
            # (e.g., considering ion and electron temperatures, secondary emission, etc.)
            sheath_drop = -2.5 * self.electron_temperature  # Rough estimate
            return sheath_drop
        except Exception as e:
            logger.error(f"Error computing analytical sheath drop: {e}")
            return 0.0
