import numpy as np
import h5py
import logging
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

class TabulatedEOS:
    """
    Tabulated Equation of State (EOS) for plasma simulations.

    This class loads EOS data from an HDF5 file and provides methods for
    interpolating thermodynamic quantities such as pressure and energy
    as functions of density and temperature.
    """

    def __init__(self, filename):
        """
        Initializes the TabulatedEOS with data from an HDF5 file.

        Args:
            filename (str): Path to the HDF5 file containing the EOS data.
        """
        try:
            with h5py.File(filename, 'r') as f:
                if not all(key in f for key in ['rho', 'T', 'p', 'e']):
                    raise ValueError("EOS table is missing required datasets.")
                self.rho_grid = f['rho'][:]
                self.T_grid = f['T'][:]
                self.p_table = f['p'][:]
                self.e_table = f['e'][:]
                if not (self.rho_grid.ndim == 1 and self.T_grid.ndim == 1 and self.p_table.ndim == 2 and self.e_table.ndim == 2):
                    raise ValueError("EOS table has incorrect dimensions.")
                if self.p_table.shape != (len(self.rho_grid), len(self.T_grid)) or self.e_table.shape != (len(self.rho_grid), len(self.T_grid)):
                    raise ValueError("EOS table has inconsistent dimensions.")
            self.p_interp = RegularGridInterpolator((self.rho_grid, self.T_grid), self.p_table)
            self.e_interp = RegularGridInterpolator((self.rho_grid, self.T_grid), self.e_table)
            logger.info(f"EOS table loaded from {filename}")
        except Exception as e:
            logger.error(f"Error loading EOS table: {e}")
            raise

    def ion_pressure(self, rho, T):
        """
        Returns the ion pressure at a given density and temperature.

        Args:
            rho (np.ndarray): Mass density (kg/m^3).
            T (np.ndarray): Temperature (K).

        Returns:
            np.ndarray: Ion pressure (Pa).
        """
        try:
            return self.p_interp(np.stack([rho, T], axis=-1))
        except Exception as e:
            logger.error(f"Error interpolating ion pressure: {e}")
            raise

    def electron_pressure(self, rho, T):
        """
        Returns the electron pressure at a given density and temperature.

        Args:
            rho (np.ndarray): Mass density (kg/m^3).
            T (np.ndarray): Temperature (K).

        Returns:
            np.ndarray: Electron pressure (Pa).
        """
        try:
            return self.p_interp(np.stack([rho, T], axis=-1))
        except Exception as e:
            logger.error(f"Error interpolating electron pressure: {e}")
            raise

    def ion_energy(self, rho, p):
        """
        Returns the ion internal energy at a given density and pressure.

        Args:
            rho (np.ndarray): Mass density (kg/m^3).
            p (np.ndarray): Pressure (Pa).

        Returns:
            np.ndarray: Ion internal energy (J/kg).
        """
        try:
            return self.e_interp(np.stack([rho, p], axis=-1))
        except Exception as e:
            logger.error(f"Error interpolating ion energy: {e}")
            raise

    def electron_energy(self, rho, p):
        """
        Returns the electron internal energy at a given density and pressure.

        Args:
            rho (np.ndarray): Mass density (kg/m^3).
            p (np.ndarray): Pressure (Pa).

        Returns:
            np.ndarray: Electron internal energy (J/kg).
        """
        try:
            return self.e_interp(np.stack([rho, p], axis=-1))
        except Exception as e:
            logger.error(f"Error interpolating electron energy: {e}")
            raise

    def __str__(self):
        """Returns a string representation of the TabulatedEOS object."""
        return f"TabulatedEOS(rho_grid={self.rho_grid.shape}, T_grid={self.T_grid.shape}, p_table={self.p_table.shape}, e_table={self.e_table.shape})"

    def __repr__(self):
        """Returns a string representation of the TabulatedEOS object."""
        return self.__str__()
