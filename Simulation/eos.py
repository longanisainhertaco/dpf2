import numpy as np
import h5py
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

class TabulatedEOS:
    """
    Tabulated Equation of State (EOS) for plasma simulations.

    This class loads EOS data from HDF5 files and provides methods for
    interpolating thermodynamic quantities such as pressure and energy
    as functions of density and temperature. When ``mixture_fractions``
    are supplied, tables for multiple species are combined into a single
    weighted EOS.
    """

    def __init__(self, filename: Union[str, Path, Dict[str, Union[str, Path]]], mixture_fractions: Optional[Dict[str, float]] = None):
        """Initializes the TabulatedEOS with data from one or more HDF5 files.

        Args:
            filename (Union[str, Path, Dict[str, Union[str, Path]]]): Path to the HDF5
                file containing the EOS data for a single species or a mapping of
                species names to file paths when ``mixture_fractions`` is supplied.
            mixture_fractions (Optional[Dict[str, float]]): Optional mixture composition
                where keys are species names and values are their fractions.
        """

        def _load_table(path: Union[str, Path]):
            with h5py.File(path, 'r') as f:
                if not all(key in f for key in ['rho', 'T', 'p', 'e']):
                    raise ValueError("EOS table is missing required datasets.")
                rho_grid = f['rho'][:]
                T_grid = f['T'][:]
                p_table = f['p'][:]
                e_table = f['e'][:]
                if not (
                    rho_grid.ndim == 1
                    and T_grid.ndim == 1
                    and p_table.ndim == 2
                    and e_table.ndim == 2
                ):
                    raise ValueError("EOS table has incorrect dimensions.")
                if p_table.shape != (len(rho_grid), len(T_grid)) or e_table.shape != (
                    len(rho_grid), len(T_grid)
                ):
                    raise ValueError("EOS table has inconsistent dimensions.")
            return rho_grid, T_grid, p_table, e_table

        try:
            if mixture_fractions:
                if isinstance(filename, (str, Path)):
                    base = Path(filename)
                    species_files = {sp: base / f"{sp}.h5" for sp in mixture_fractions}
                elif isinstance(filename, dict):
                    species_files = {sp: Path(path) for sp, path in filename.items()}
                else:
                    raise TypeError("filename must be a path or mapping when mixture_fractions are provided")

                first = True
                for species, path in species_files.items():
                    rho, T, p_tab, e_tab = _load_table(path)
                    weight = mixture_fractions.get(species, 0.0)
                    if first:
                        self.rho_grid = rho
                        self.T_grid = T
                        self.p_table = weight * p_tab
                        self.e_table = weight * e_tab
                        first = False
                    else:
                        if not (
                            np.array_equal(self.rho_grid, rho)
                            and np.array_equal(self.T_grid, T)
                        ):
                            raise ValueError("EOS grids for different species do not match.")
                        self.p_table += weight * p_tab
                        self.e_table += weight * e_tab
                logger.info("Mixture EOS tables loaded for species: %s", ", ".join(species_files.keys()))
            else:
                rho, T, p_tab, e_tab = _load_table(filename)
                self.rho_grid = rho
                self.T_grid = T
                self.p_table = p_tab
                self.e_table = e_tab
                logger.info(f"EOS table loaded from {filename}")
            self.p_interp = RegularGridInterpolator((self.rho_grid, self.T_grid), self.p_table)
            self.e_interp = RegularGridInterpolator((self.rho_grid, self.T_grid), self.e_table)
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
