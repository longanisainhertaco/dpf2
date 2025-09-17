import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

try:  # pragma: no cover - optional SciPy dependency
    from scipy.interpolate import RegularGridInterpolator
except ModuleNotFoundError:  # pragma: no cover

    class RegularGridInterpolator:  # type: ignore[misc]
        """Very small fallback interpolator when SciPy is unavailable."""

        def __init__(self, points, values):
            self.x, self.y = [np.array(p) for p in points]
            self.values = np.array(values)

        def __call__(self, pts):  # noqa: D401 - behave like SciPy callable
            result = []
            for x, y in pts:
                i = np.searchsorted(self.x, x, side="right") - 1
                j = np.searchsorted(self.y, y, side="right") - 1
                i = max(min(i, len(self.x) - 2), 0)
                j = max(min(j, len(self.y) - 2), 0)
                x0, x1 = self.x[i], self.x[i + 1]
                y0, y1 = self.y[j], self.y[j + 1]
                tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
                ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
                f00 = self.values[i, j]
                f01 = self.values[i, j + 1]
                f10 = self.values[i + 1, j]
                f11 = self.values[i + 1, j + 1]
                val = (
                    f00 * (1 - tx) * (1 - ty)
                    + f01 * (1 - tx) * ty
                    + f10 * tx * (1 - ty)
                    + f11 * tx * ty
                )
                result.append(val)
            return np.array(result)


try:  # optional dependency
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError("h5py is required; install dpf2[warpx]") from exc

logger = logging.getLogger(__name__)


def parse_mixture_fractions(
    mixture_fractions: Union[str, Dict[str, float], None],
) -> Dict[str, float]:
    """Parse mixture fraction definitions into a normalised dictionary.

    Parameters
    ----------
    mixture_fractions:
        Either a mapping of species to fractions or a comma separated string of
        ``"species:fraction"`` pairs. Fractions must be non-negative and sum to
        one. ``None`` returns an empty dictionary.
    """

    if mixture_fractions is None:
        return {}

    parsed: Dict[str, float] = {}

    if isinstance(mixture_fractions, str):
        for part in mixture_fractions.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                species, frac = part.split(":")
            except ValueError as exc:
                raise ValueError(
                    "Mixture fractions must be in 'species:fraction' format"
                ) from exc
            species = species.strip()
            if not species:
                raise ValueError("Species name in mixture fractions cannot be empty")
            try:
                parsed[species] = float(frac)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid fraction value for species '{species}': {frac}"
                ) from exc
    elif isinstance(mixture_fractions, dict):
        for species, frac in mixture_fractions.items():
            species = str(species)
            try:
                parsed[species] = float(frac)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid fraction value for species '{species}': {frac}"
                ) from exc
    else:
        raise TypeError("mixture_fractions must be a dict or a string")

    if not parsed:
        raise ValueError("No valid mixture fractions provided")

    if any(frac < 0.0 for frac in parsed.values()):
        raise ValueError("Mixture fractions must be non-negative")

    total = sum(parsed.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Mixture fractions must sum to 1")

    return parsed


class TabulatedEOS:
    """
    Tabulated Equation of State (EOS) for plasma simulations.

    This class loads EOS data from HDF5 files and provides methods for
    interpolating thermodynamic quantities such as pressure and energy
    as functions of density and temperature. When ``mixture_fractions``
    are supplied, tables for multiple species are combined into a single
    weighted EOS.
    """

    def __init__(
        self,
        filename: Union[str, Path, Dict[str, Union[str, Path]]],
        mixture_fractions: Optional[Union[str, Dict[str, float]]] = None,
    ) -> None:
        """Initializes the TabulatedEOS with data from one or more HDF5 files.

        Args:
            filename (Union[str, Path, Dict[str, Union[str, Path]]]): Path to the HDF5
                file containing the EOS data for a single species or a mapping of
                species names to file paths when ``mixture_fractions`` is supplied.
            mixture_fractions (Optional[Union[str, Dict[str, float]]]): Optional
                mixture composition. Can be provided as a dictionary or as a comma
                separated string of ``"species:fraction"`` pairs.
        """

        def _load_table(path: Union[str, Path]):
            with h5py.File(path, "r") as f:
                if not all(key in f for key in ["rho", "T", "p", "e"]):
                    raise ValueError("EOS table is missing required datasets.")
                rho_grid = f["rho"][:]
                T_grid = f["T"][:]
                p_table = f["p"][:]
                e_table = f["e"][:]
                if not (
                    rho_grid.ndim == 1
                    and T_grid.ndim == 1
                    and p_table.ndim == 2
                    and e_table.ndim == 2
                ):
                    raise ValueError("EOS table has incorrect dimensions.")
                if p_table.shape != (len(rho_grid), len(T_grid)) or e_table.shape != (
                    len(rho_grid),
                    len(T_grid),
                ):
                    raise ValueError("EOS table has inconsistent dimensions.")
            return rho_grid, T_grid, p_table, e_table

        fractions = parse_mixture_fractions(mixture_fractions)
        self.mixture_fractions = fractions

        try:
            if fractions:
                if isinstance(filename, (str, Path)):
                    base = Path(filename)
                    species_files = {sp: base / f"{sp}.h5" for sp in fractions}
                elif isinstance(filename, dict):
                    missing = set(fractions) - set(filename)
                    if missing:
                        raise ValueError(
                            "Missing EOS data for species: "
                            + ", ".join(sorted(missing))
                        )
                    species_files = {sp: Path(filename[sp]) for sp in fractions}
                else:
                    raise TypeError(
                        "filename must be a path or mapping when mixture_fractions are provided"
                    )

                missing_files = [
                    sp for sp, path in species_files.items() if not path.is_file()
                ]
                if missing_files:
                    raise ValueError(
                        "Missing EOS data for species: "
                        + ", ".join(sorted(missing_files))
                    )

                for idx, (species, path) in enumerate(species_files.items()):
                    rho, T, p_tab, e_tab = _load_table(path)
                    weight = fractions.get(species, 0.0)
                    if idx == 0:
                        self.rho_grid = rho
                        self.T_grid = T
                        self.p_table = weight * p_tab
                        self.e_table = weight * e_tab
                    else:
                        if not (
                            np.array_equal(self.rho_grid, rho)
                            and np.array_equal(self.T_grid, T)
                        ):
                            raise ValueError(
                                "EOS grids for different species do not match."
                            )
                        self.p_table += weight * p_tab
                        self.e_table += weight * e_tab
                logger.info(
                    "Mixture EOS tables loaded for species: %s",
                    ", ".join(species_files.keys()),
                )
            else:
                rho, T, p_tab, e_tab = _load_table(filename)
                self.rho_grid = rho
                self.T_grid = T
                self.p_table = p_tab
                self.e_table = e_tab
                logger.info(f"EOS table loaded from {filename}")
            self.p_interp = RegularGridInterpolator(
                (self.rho_grid, self.T_grid), self.p_table
            )
            self.e_interp = RegularGridInterpolator(
                (self.rho_grid, self.T_grid), self.e_table
            )
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

    def ion_energy(self, rho, T):
        """Return ion internal energy for the given density and temperature.

        Parameters
        ----------
        rho: np.ndarray
            Mass density (kg/m^3).
        T: np.ndarray
            Temperature (K).

        Returns
        -------
        np.ndarray
            Ion specific internal energy (J/kg).

        """
        try:
            return self.e_interp(np.stack([rho, T], axis=-1))
        except Exception as e:
            logger.error(f"Error interpolating ion energy: {e}")
            raise

    def electron_energy(self, rho, T):
        """Return electron internal energy for the given density and temperature."""
        try:
            return self.e_interp(np.stack([rho, T], axis=-1))
        except Exception as e:
            logger.error(f"Error interpolating electron energy: {e}")
            raise

    def __str__(self):
        """Returns a string representation of the TabulatedEOS object."""
        return (
            f"TabulatedEOS(rho_grid={self.rho_grid.shape}, "
            f"T_grid={self.T_grid.shape}, "
            f"p_table={self.p_table.shape}, e_table={self.e_table.shape})"
        )

    def __repr__(self):
        """Returns a string representation of the TabulatedEOS object."""
        return self.__str__()
