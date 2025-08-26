import numpy as np
import logging

# Optional heavy imports ----------------------------------------------------
# Many of the production dependencies (matplotlib, scipy, numba) are not
# required for the lightweight behaviour exercised in the tests.  To keep the
# module importable in minimal environments we guard these imports and provide
# fallbacks where possible.
try:  # pragma: no cover - exercised implicitly during import
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib is not needed in tests
    plt = None  # noqa: F401

try:  # pragma: no cover - used only by the high-fidelity model
    from scipy.optimize import root_scalar  # type: ignore
    from scipy.linalg import solve  # type: ignore
    from scipy.sparse import diags  # type: ignore
except Exception:  # pragma: no cover - scipy not available
    root_scalar = solve = diags = None  # type: ignore

try:  # pragma: no cover - numerical acceleration optional
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover - numba not available
    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator

    def prange(*args, **kwargs):  # type: ignore
        return range(*args)

from typing import Dict, Any

from .models import PhysicsModule, SimulationState  # type: ignore

try:  # pragma: no cover - config schema depends on pydantic
    from .config_schema import SheathConfig  # type: ignore
except Exception:  # pragma: no cover - when pydantic v2 not present
    SheathConfig = Any  # type: ignore

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


class BohmSheath:
    """Apply simple Bohm sheath boundary conditions.

    Only a very small subset of the full sheath physics is required for the
    unit tests.  This implementation computes the Bohm velocity and applies it
    either to a ``FieldManager``/``SimulationState`` electric field or to raw
    density/momentum arrays.  This replaces the previous no-op placeholder so
    that legacy components relying on a ``BohmSheath`` interface exhibit
    physically motivated behaviour in tests.
    """

    def __init__(self,
                 geometry: Any | None = None,
                 electron_temperature: float = 1.0,
                 ion_mass: float = 1.6726219e-27,
                 axis: int = 2) -> None:
        """Create a Bohm sheath model.

        Parameters
        ----------
        geometry: Any, optional
            Placeholder for geometric information (unused but preserved for
            API compatibility).
        electron_temperature: float, optional
            Electron temperature in eV used to compute the Bohm velocity.
        ion_mass: float, optional
            Ion mass in kg.
        axis: int, optional
            Axis normal to the sheath; default corresponds to the ``z`` axis.
        """

        self.geometry = geometry
        self.electron_temperature = electron_temperature
        self.ion_mass = ion_mass
        self.axis = axis

        # Expose last computed sheath properties for diagnostics/testing
        self.last_velocity = 0.0
        self.last_potential = 0.0

    # ------------------------------------------------------------------
    def _bohm_velocity(self) -> float:
        """Return the Bohm velocity based on the stored temperature/mass."""
        Te_joule = self.electron_temperature * e_charge
        return np.sqrt(Te_joule / self.ion_mass)

    # ------------------------------------------------------------------
    def _sheath_potential(self) -> float:
        r"""Compute the sheath potential drop using the Bohm criterion.

        A very simple estimate for the potential drop at a floating wall is
        obtained by equating the ion and electron fluxes, yielding

        .. math::

            \phi_s = T_e \ln\sqrt{m_i/(2\pi m_e)}.

        The electron temperature ``T_e`` is provided in electron-volts, so the
        resulting potential is also in volts.
        """

        mass_ratio = self.ion_mass / (2 * np.pi * m_e)
        return self.electron_temperature * np.log(np.sqrt(mass_ratio))

    # ------------------------------------------------------------------
    def _apply_to_field_manager(self, fm: Any, sheath_field: float) -> None:
        """Apply boundary condition directly to a ``FieldManager``.

        The electric field array ``E`` stored in the :class:`FieldManager` has
        the shape ``(3, Nx, Ny, Nz)`` with the first index corresponding to the
        field component.  Only the component normal to the sheath and the cells
        adjacent to the boundary should be modified.  ``axis`` indicates both
        which field component to touch and which spatial dimension's high-side
        boundary to update.
        """

        # Take a defensive copy so callers holding a reference to the
        # ``FieldManager``'s electric field do not see it mutate until the
        # update is committed.  This mirrors the behaviour of the public
        # :meth:`update_E` API which expects a full array to be supplied.
        E = fm.get_E().copy()

        # Build an index tuple selecting the component normal to the sheath
        # (``self.axis``) and the high-side boundary cell along the same
        # spatial direction.  All transverse indices are left as ``slice(None)``
        # so that the entire boundary plane is modified in one operation.
        idx = [self.axis, slice(None), slice(None), slice(None)]
        idx[self.axis + 1] = -1
        E[tuple(idx)] = sheath_field

        fm.update_E(E)

    # ------------------------------------------------------------------
    def apply(self, target: Any, momentum: Any | None = None) -> None:
        """Apply Bohm sheath boundary conditions.

        Parameters
        ----------
        target:
            One of ``SimulationState``, ``FieldManager`` or a density array.
        momentum: array-like, optional
            Momentum array required when ``target`` is a density field.
        """

        v_bohm = self._bohm_velocity()
        phi_s = self._sheath_potential()

        # Record the quantities used for boundary updates so that callers can
        # inspect them after application.  This is helpful for debugging and
        # for unit tests that wish to verify the Bohm criterion is enforced.
        self.last_velocity = v_bohm
        self.last_potential = phi_s

        # Determine grid spacing along the sheath-normal axis for field estimate
        spacing = 1.0
        if self.axis == 0:
            if hasattr(target, 'dx'):
                spacing = getattr(target, 'dx', 1.0)
            elif hasattr(target, 'field_manager') and hasattr(target.field_manager, 'dx'):
                spacing = target.field_manager.dx
        elif self.axis == 1:
            if hasattr(target, 'dy'):
                spacing = getattr(target, 'dy', 1.0)
            elif hasattr(target, 'field_manager') and hasattr(target.field_manager, 'dy'):
                spacing = target.field_manager.dy
        else:
            if hasattr(target, 'dz'):
                spacing = getattr(target, 'dz', 1.0)
            elif hasattr(target, 'field_manager') and hasattr(target.field_manager, 'dz'):
                spacing = target.field_manager.dz
        sheath_field = phi_s / spacing

        # Case 1: target is a SimulationState holding a FieldManager
        if hasattr(target, "field_manager"):
            fm = getattr(target, "field_manager", None)
            if fm is not None:
                self._apply_to_field_manager(fm, sheath_field)

            # Update velocity field if present or create one
            vel = getattr(target, 'velocity', None)
            if vel is None:
                vel = np.zeros((3,) + target.grid_shape)
                target.velocity = vel
            if self.axis == 0:
                vel[0, -1, :, :] = v_bohm
            elif self.axis == 1:
                vel[1, :, -1, :] = v_bohm
            else:
                vel[2, :, :, -1] = v_bohm

            # Update electrostatic potential field
            phi = getattr(target, 'potential', None)
            if phi is None:
                phi = np.zeros(target.grid_shape)
                target.potential = phi
            if self.axis == 0:
                phi[-1, :, :] = phi_s
            elif self.axis == 1:
                phi[:, -1, :] = phi_s
            else:
                phi[:, :, -1] = phi_s
            return

        # Case 2: target is itself a FieldManager
        if hasattr(target, "get_E") and hasattr(target, "update_E"):
            self._apply_to_field_manager(target, sheath_field)
            return

        # Case 3: raw arrays -- modify the momentum to satisfy Bohm velocity
        density = target
        if momentum is None:
            logger.warning("Momentum array required when applying BohmSheath to raw arrays.")
            return

        try:
            if self.axis == 0:
                momentum[0, -1, :, :] = density[-1, :, :] * v_bohm
            elif self.axis == 1:
                momentum[1, :, -1, :] = density[:, -1, :] * v_bohm
            else:
                momentum[2, :, :, -1] = density[:, :, -1] * v_bohm
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.error(f"Failed to apply Bohm sheath to arrays: {exc}")

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
        # Optional parameters with sensible defaults so that lightweight test
        # configurations can omit them without failing initialisation.
        self.secondary_emission_coefficient = getattr(config, "secondary_emission_coefficient", 0.0)
        self.electron_distribution = getattr(config, "electron_distribution", "maxwellian")
        self.electron_distribution_params = getattr(config, "electron_distribution_params", {})

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
        """Compute electron flux using a potentially non-Maxwellian distribution."""
        try:
            if self.electron_distribution == "analytic":
                dist_fn = self.electron_distribution_params.get("distribution_fn")
                if dist_fn is None:
                    raise ValueError("distribution_fn must be provided for analytic distribution")
                v_max = self.electron_distribution_params.get("v_max", 1e7)
                num = self.electron_distribution_params.get("num_points", 1000)
                v = np.linspace(0.0, v_max, num)
                f = dist_fn(v)
                # Normalize distribution to unity density
                density_norm = np.trapz(4 * np.pi * v**2 * f, v)
                if density_norm <= 0:
                    return 0.0
                flux = np.pi * np.trapz(v**3 * f, v)
                return self.electron_density * flux / density_norm
            else:
                # Default Maxwellian result
                v_th = np.sqrt(8 * e_charge * self.electron_temperature / (np.pi * m_e))
                return 0.25 * self.electron_density * v_th
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
        """Estimate the sheath potential drop using flux balance."""
        try:
            Te = self.electron_temperature
            Ti = self.ion_temperature
            mi = self.ion_mass
            delta = getattr(self, "secondary_emission_coefficient", 0.0)

            c_s = np.sqrt(e_charge * (Te + Ti) / mi)
            v_th = np.sqrt(8 * e_charge * Te / (np.pi * m_e))
            sheath_drop = Te * np.log((1 - delta) * 4 * c_s / v_th)
            return sheath_drop
        except Exception as e:
            logger.error(f"Error computing analytical sheath drop: {e}")
            return 0.0
