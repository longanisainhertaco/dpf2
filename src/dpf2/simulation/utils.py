# utils.py
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FieldManager:
    """
    Manages electromagnetic fields (E and B) and related operations.
    """

    def __init__(self,
                 grid_shape: Tuple[int, int, int],
                 dx: float,
                 dy: float,
                 dz: float,
                 domain_lo: Tuple[float, float, float],
                 boundary_conditions: Dict[str, str]):
        """
        Initializes the FieldManager with grid parameters and boundary conditions.

        Args:
            grid_shape: The shape of the grid (nx, ny, nz).
            dx: The grid spacing in the x-direction.
            dy: The grid spacing in the y-direction.
            dz: The grid spacing in the z-direction.
            domain_lo: The lower corner of the domain (x0, y0, z0).
            boundary_conditions: A dictionary of boundary conditions.
        """
        self.nx, self.ny, self.nz = grid_shape
        self.dx, self.dy, self.dz = dx, dy, dz
        self.domain_lo = domain_lo
        self.boundary_conditions = boundary_conditions

        # Initialize fields
        self.E = np.zeros((3, self.nx, self.ny, self.nz))  # Electric field (3 components)
        self.B = np.zeros((3, self.nx, self.ny, self.nz))  # Magnetic field (3 components)
        self.J = np.zeros((3, self.nx, self.ny, self.nz))  # Current density (3 components)
        # Charge density
        self.rho = np.zeros((self.nx, self.ny, self.nz))  # Charge density

        logger.info("FieldManager initialized.")

    def update_E(self, new_E: np.ndarray):
        """Updates the electric field."""
        if new_E.shape != self.E.shape:
            raise ValueError(f"Invalid shape for E: expected {self.E.shape}, got {new_E.shape}")
        self.E = new_E

    def update_B(self, new_B: np.ndarray):
        """Updates the magnetic field."""
        if new_B.shape != self.B.shape:
            raise ValueError(f"Invalid shape for B: expected {self.B.shape}, got {new_B.shape}")
        self.B = new_B

    def get_E(self) -> np.ndarray:
        """Returns the electric field."""
        return self.E

    def get_B(self) -> np.ndarray:
        """Returns the magnetic field."""
        return self.B

    def get_J(self) -> np.ndarray:
        """Returns the current density."""
        return self.J

    # ------------------------------------------------------------------
    # Finite-difference utilities
    # ------------------------------------------------------------------
    def _yee_derivative(self,
                        f: np.ndarray,
                        axis: int,
                        h: float,
                        bc_lo: str,
                        bc_hi: str) -> np.ndarray:
        """Compute a Yee-style derivative along a given axis.

        A central difference is used in the interior while the
        boundaries honour the specified boundary conditions.

        Args:
            f: Field component to differentiate.
            axis: Axis along which to take the derivative.
            h: Grid spacing for the axis.
            bc_lo: Boundary condition on the low side ('periodic',
                'neumann', 'dirichlet', or 'outflow').
            bc_hi: Boundary condition on the high side.

        Returns:
            Array of the same shape as ``f`` containing df/dx_axis.
        """

        # Start with periodic central differences using ``np.roll``
        df = (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * h)

        # Helper to index first/last cell along ``axis``
        def slc(idx):
            s = [slice(None)] * f.ndim
            s[axis] = idx
            return tuple(s)

        # Low-side boundary
        if bc_lo == 'dirichlet':
            df[slc(0)] = (f[slc(1)] - 0.0) / h
        elif bc_lo in {'neumann', 'outflow'}:
            df[slc(0)] = (f[slc(1)] - f[slc(0)]) / h

        # High-side boundary
        if bc_hi == 'dirichlet':
            df[slc(-1)] = (0.0 - f[slc(-2)]) / h
        elif bc_hi in {'neumann', 'outflow'}:
            df[slc(-1)] = (f[slc(-1)] - f[slc(-2)]) / h

        return df

    def compute_divergence(self, field: np.ndarray) -> np.ndarray:
        """Computes the divergence of a field using Yee-style differences."""
        bx_lo = self.boundary_conditions.get('x_lo', 'periodic')
        bx_hi = self.boundary_conditions.get('x_hi', 'periodic')
        by_lo = self.boundary_conditions.get('y_lo', 'periodic')
        by_hi = self.boundary_conditions.get('y_hi', 'periodic')
        bz_lo = self.boundary_conditions.get('z_lo', 'periodic')
        bz_hi = self.boundary_conditions.get('z_hi', 'periodic')

        div_x = self._yee_derivative(field[0], 0, self.dx, bx_lo, bx_hi)
        div_y = self._yee_derivative(field[1], 1, self.dy, by_lo, by_hi)
        div_z = self._yee_derivative(field[2], 2, self.dz, bz_lo, bz_hi)

        return div_x + div_y + div_z

    def compute_curl(self, field: np.ndarray) -> np.ndarray:
        """Computes the curl of a field using Yee-style differences."""
        bx_lo = self.boundary_conditions.get('x_lo', 'periodic')
        bx_hi = self.boundary_conditions.get('x_hi', 'periodic')
        by_lo = self.boundary_conditions.get('y_lo', 'periodic')
        by_hi = self.boundary_conditions.get('y_hi', 'periodic')
        bz_lo = self.boundary_conditions.get('z_lo', 'periodic')
        bz_hi = self.boundary_conditions.get('z_hi', 'periodic')

        dFz_dy = self._yee_derivative(field[2], 1, self.dy, by_lo, by_hi)
        dFy_dz = self._yee_derivative(field[1], 2, self.dz, bz_lo, bz_hi)
        dFx_dz = self._yee_derivative(field[0], 2, self.dz, bz_lo, bz_hi)
        dFz_dx = self._yee_derivative(field[2], 0, self.dx, bx_lo, bx_hi)
        dFy_dx = self._yee_derivative(field[1], 0, self.dx, bx_lo, bx_hi)
        dFx_dy = self._yee_derivative(field[0], 1, self.dy, by_lo, by_hi)

        curl_x = dFz_dy - dFy_dz
        curl_y = dFx_dz - dFz_dx
        curl_z = dFy_dx - dFx_dy

        return np.array([curl_x, curl_y, curl_z])

    def apply_boundary_conditions(self, state: "SimulationState"):
        """Applies boundary conditions to the fields.

        Supports standard boundary types (periodic, Neumann, Dirichlet, outflow)
        as well as a simple perfectly matched layer (PML).  The PML damps field
        components inside the ghost region according to configurable parameters:

        ``pml_thickness``
            Number of ghost cells over which to apply damping (default: 2).

        ``pml_sigma``
            Damping coefficient for the exponential profile (default: 2.0).

        ``pml_profile``
            Damping profile name.  Currently ``"exponential"`` (default) and
            ``"linear"`` are recognised.
        """

        g = int(self.boundary_conditions.get("ghost_cells", 2))

        def _pml_factors(thickness: int, sigma: float, profile: str) -> np.ndarray:
            if profile == "linear":
                factors = 1.0 - (np.arange(1, thickness + 1) / thickness) * sigma
                return np.clip(factors, 0.0, 1.0)
            # Default to exponential decay
            return np.exp(-sigma * np.arange(1, thickness + 1))

        # Helper function to apply boundary conditions to a field
        def apply_bc(field, bc_key, axis):
            bc = self.boundary_conditions.get(bc_key, 'periodic')
            side = 'lo' if bc_key.endswith('lo') else 'hi'
            if bc == 'periodic':
                # Periodic boundary conditions
                field = np.roll(field, shift=g, axis=axis)
            elif bc == 'neumann':
                # Neumann (zero-gradient) boundary conditions
                if axis == 0:
                    field[:g, :, :] = field[g:2 * g, :, :]
                    field[-g:, :, :] = field[-2 * g:-g, :, :]
                elif axis == 1:
                    field[:, :g, :] = field[:, g:2 * g, :]
                    field[:, -g:, :] = field[:, -2 * g:-g, :]
                elif axis == 2:
                    field[:, :, :g] = field[:, :, g:2 * g]
                    field[:, :, -g:] = field[:, :, -2 * g:-g]
            elif bc == 'dirichlet':
                # Dirichlet (fixed value) boundary conditions - set to zero for simplicity
                if axis == 0:
                    field[:g, :, :] = 0.0
                    field[-g:, :, :] = 0.0
                elif axis == 1:
                    field[:, :g, :] = 0.0
                    field[:, -g:, :] = 0.0
                elif axis == 2:
                    field[:, :, :g] = 0.0
                    field[:, :, -g:] = 0.0
            elif bc == 'outflow':
                # Outflow boundary conditions (copy values from the interior)
                if axis == 0:
                    field[:g, :, :] = field[g:2 * g, :, :]
                    field[-g:, :, :] = field[-2 * g:-g, :, :]
                elif axis == 1:
                    field[:, :g, :] = field[:, g:2 * g, :]
                    field[:, -g:, :] = field[:, -2 * g:-g, :]
                elif axis == 2:
                    field[:, :, :g] = field[:, :, g:2 * g]
                    field[:, :, -g:] = field[:, :, -2 * g:-g]
            elif bc == 'pml':
                # Simple PML implemented via damping in ghost cells
                thickness = int(self.boundary_conditions.get('pml_thickness', g))
                thickness = min(thickness, g)
                sigma = float(self.boundary_conditions.get('pml_sigma', 2.0))
                profile = self.boundary_conditions.get('pml_profile', 'exponential')
                factors = _pml_factors(thickness, sigma, profile)

                def _apply(axis_slc, interior):
                    facs = factors[::-1] if side == 'lo' else factors
                    if axis == 0:
                        facs = facs[:, None, None]
                    elif axis == 1:
                        facs = facs[None, :, None]
                    else:
                        facs = facs[None, None, :]
                    field[axis_slc] = interior * facs

                if axis == 0:
                    interior_idx = thickness if side == 'lo' else -thickness - 1
                    interior = np.copy(field[interior_idx, :, :])[None, :, :]
                    sl = slice(0, thickness) if side == 'lo' else slice(-thickness, None)
                    _apply((sl, slice(None), slice(None)), interior)
                    if thickness < g:
                        zero_sl = slice(thickness, g) if side == 'lo' else slice(-g, -thickness)
                        field[zero_sl, :, :] = 0.0
                elif axis == 1:
                    interior_idx = thickness if side == 'lo' else -thickness - 1
                    interior = np.copy(field[:, interior_idx, :])[:, None, :]
                    sl = slice(0, thickness) if side == 'lo' else slice(-thickness, None)
                    _apply((slice(None), sl, slice(None)), interior)
                    if thickness < g:
                        zero_sl = slice(thickness, g) if side == 'lo' else slice(-g, -thickness)
                        field[:, zero_sl, :] = 0.0
                elif axis == 2:
                    interior_idx = thickness if side == 'lo' else -thickness - 1
                    interior = np.copy(field[:, :, interior_idx])[:, :, None]
                    sl = slice(0, thickness) if side == 'lo' else slice(-thickness, None)
                    _apply((slice(None), slice(None), sl), interior)
                    if thickness < g:
                        zero_sl = slice(thickness, g) if side == 'lo' else slice(-g, -thickness)
                        field[:, :, zero_sl] = 0.0
            else:
                logger.warning(f"Unknown boundary condition: {bc} - defaulting to periodic.")
                field = np.roll(field, shift=g, axis=axis)
            return field

        # Apply boundary conditions to each field component in-place without
        # creating temporary stacked arrays.  This ensures that ``E``, ``B`` and
        # ``J`` maintain their original shapes and avoids the overhead of
        # re-stacking arrays after every boundary update.
        for i in range(3):
            for axis_num, axis in enumerate(["x", "y", "z"]):
                for side in ["lo", "hi"]:
                    apply_bc(self.E[i], f"{axis}_{side}", axis_num)
                    apply_bc(self.B[i], f"{axis}_{side}", axis_num)
                    apply_bc(self.J[i], f"{axis}_{side}", axis_num)

        # Verify field shapes remain intact after boundary application
        expected_shape = (3, self.nx, self.ny, self.nz)
        for name, field in (('E', self.E), ('B', self.B), ('J', self.J)):
            if field.shape != expected_shape:
                raise ValueError(f"{name} has invalid shape {field.shape}; expected {expected_shape}")

        # Coordinate with sheath and circuit BCs (example - needs adaptation)
        # if self.sheath_model:
        #     sheath_potential = self.sheath_model.get_sheath_potential()
        #     # Apply sheath potential as a boundary condition on E field
        #     self.E[0, -g:, :, :] = sheath_potential  # Example: Apply to x-component at high boundary

        logger.debug("Boundary conditions applied to fields.")

    def deposit_charge(self, charge_density: np.ndarray):
        """Deposits charge from particles to the grid."""
        if charge_density.shape != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"Invalid shape for charge_density: expected {(self.nx, self.ny, self.nz)}, got {charge_density.shape}"
            )
        self.rho += charge_density

    def deposit_current(self, current_density: np.ndarray):
        """Deposits current from particles to the grid."""
        if current_density.shape != self.J.shape:
            raise ValueError(f"Invalid shape for current_density: expected {self.J.shape}, got {current_density.shape}")
        self.J += current_density

    def initialize(self):
        """Initializes the fields."""
        logger.info("FieldManager initialized.")

    def finalize(self):
        """Finalizes the fields."""
        logger.info("FieldManager finalized.")

    def checkpoint(self) -> Dict[str, Any]:
        """Returns a dictionary of data to checkpoint."""
        try:
            checkpoint_data = {
                'E': self.E.tolist(),
                'B': self.B.tolist(),
                'J': self.J.tolist(),
                'rho': self.rho.tolist()
            }
            return checkpoint_data
        except Exception as e:
            logger.error(f"Error during checkpoint: {e}")
            return {}

    def restart(self, data: Dict[str, Any]):
        """Restores data from a checkpoint."""
        try:
            self.E = np.array(data.get('E', self.E))
            self.B = np.array(data.get('B', self.B))
            self.J = np.array(data.get('J', self.J))
            self.rho = np.array(data.get('rho', self.rho))
        except Exception as e:
            logger.error(f"Error during restart: {e}")

    def configure(self, config: Dict[str, Any]):
        """Configures the field manager."""
        try:
            for key, value in config.items():
                setattr(self, key, value)
            logger.info(f"FieldManager configured with: {config}")
        except Exception as e:
            logger.error(f"Error configuring FieldManager: {e}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns diagnostic information."""
        try:
            div_B = self.compute_divergence(self.B)
            max_div_B = np.max(np.abs(div_B))
            return {
                "max_div_B": max_div_B,
                "rho": self.rho.tolist(),
                # Add other diagnostics as needed
            }
        except Exception as e:
            logger.error(f"Error getting diagnostics: {e}")
            return {}

class SimulationState:
    """
    Represents the state of the simulation at a given time.

    This class encapsulates all the data that describes the current state
    of the simulation, including physical quantities like density, velocity,
    temperature, and fields.
    """

    def __init__(self,
                 grid_shape: Tuple[int, int, int],
                 dx: float,
                 dy: float,
                 dz: float,
                 domain_lo: Tuple[float, float, float],
                 boundary_conditions: Dict[str, str],
                 density: Optional[np.ndarray] = None,
                 velocity: Optional[np.ndarray] = None,
                 pressure: Optional[np.ndarray] = None,
                 internal_energy: Optional[np.ndarray] = None,
                 electron_temperature: Optional[np.ndarray] = None,
                 ion_temperature: Optional[np.ndarray] = None,
                 current_density: Optional[np.ndarray] = None,
                 neutral_density: Optional[np.ndarray] = None,
                 ion_density: Optional[np.ndarray] = None,
                 viscosity: Optional[np.ndarray] = None,
                 time: Optional[float] = 0.0,
                 **kwargs):
        """
        Initializes the SimulationState with optional data arrays.

        Args:
            grid_shape: The shape of the grid (nx, ny, nz).
            dx: The grid spacing in the x-direction.
            dy: The grid spacing in the y-direction.
            dz: The grid spacing in the z-direction.
            domain_lo: The lower corner of the domain (x0, y0, z0).
            boundary_conditions: A dictionary of boundary conditions.
            density: The mass density field.
            velocity: The velocity field (3D vector).
            pressure: The pressure field.
            internal_energy: The internal energy field.
            electron_temperature: The electron temperature field.
            ion_temperature: The ion temperature field.
            current_density: The current density field.
            neutral_density: The neutral density field.
            ion_density: The ion density field.
            viscosity: The viscosity field.
            time: The simulation time.
            **kwargs: Additional keyword arguments for custom fields.
        """
        self.grid_shape = grid_shape
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.domain_lo = domain_lo
        self.boundary_conditions = boundary_conditions
        self.density = density if density is not None else np.zeros(grid_shape) # Ensure density is initialized
        self.velocity = velocity
        self.pressure = pressure
        self.internal_energy = internal_energy
        self.electron_temperature = electron_temperature
        self.ion_temperature = ion_temperature
        self.current_density = current_density
        self.neutral_density = neutral_density
        self.ion_density = ion_density
        self.viscosity = viscosity
        self.time = time

        # Add other fields as needed
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        """Returns a string representation of the SimulationState."""
        return f"SimulationState(density={self.density.shape if self.density is not None else None}, " \
               f"velocity={self.velocity.shape if self.velocity is not None else None}, " \
               f"pressure={self.pressure.shape if self.pressure is not None else None}, " \
               f"internal_energy={self.internal_energy.shape if self.internal_energy is not None else None}, " \
               f"electron_temperature={self.electron_temperature.shape if self.electron_temperature is not None else None}, " \
               f"ion_temperature={self.ion_temperature.shape if self.ion_temperature is not None else None}, " \
               f"current_density={self.current_density.shape if self.current_density is not None else None}, " \
               f"neutral_density={self.neutral_density.shape if self.neutral_density is not None else None}, " \
               f"ion_density={self.ion_density.shape if self.ion_density is not None else None}, " \
               f"viscosity={self.viscosity.shape if self.viscosity is not None else None}, " \
               f"time={self.time})"

    def __repr__(self):
        """Returns a string representation of the SimulationState."""
        return self.__str__()
