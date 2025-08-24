from __future__ import annotations

"""Equation of state (EOS) models used by the DPF solver.

This module provides a minimal interface for pressure and energy
calculations.  It now supports tabulated equations of state that supply
pressure and specific internal energy as functions of density and
temperature, including multi-species real-gas mixtures built from
individual species tables.  Interpolation is performed using SciPy's
``RegularGridInterpolator`` which yields bilinear behaviour on a regular
grid.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

try:  # pragma: no cover - optional SciPy dependency
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import root_scalar
except ModuleNotFoundError:  # pragma: no cover
    class RegularGridInterpolator:  # type: ignore[misc]
        def __init__(self, *args, **kwargs):  # noqa: D401 - simple stub
            raise ModuleNotFoundError("SciPy is required for interpolation")

    def root_scalar(*args, **kwargs):  # type: ignore[misc]
        raise ModuleNotFoundError("SciPy is required for root finding")

try:
    from ..core_schema import EOSModel
except Exception:  # pragma: no cover - minimal fallback without pydantic
    from enum import Enum

    class EOSModel(str, Enum):
        IDEAL = "ideal"
        TABULATED = "tabulated"
        REAL_GAS = "real_gas"

try:  # pragma: no cover - fallback when simulation EOS is unavailable
    from dpf2.simulation.eos import TabulatedEOS as _SimulationTabulatedEOS
except Exception:  # pragma: no cover
    import json
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        h5py = None  # type: ignore

    class _SimulationTabulatedEOS:  # type: ignore[misc]
        def __init__(self, filename, mixture_fractions=None):
            import numpy as np

            def load(path):
                path = Path(path)
                if path.suffix == ".json":
                    with open(path, "r", encoding="utf8") as f:
                        data = json.load(f)
                    return (
                        np.array(data["rho"]),
                        np.array(data["T"]),
                        np.array(data["p"]),
                        np.array(data["e"]),
                    )
                if h5py is None:
                    raise ModuleNotFoundError("h5py is required for tabulated EOS")
                with h5py.File(path, "r") as f:
                    return f["rho"][:], f["T"][:], f["p"][:], f["e"][:]

            if mixture_fractions:
                if isinstance(filename, (str, Path)):
                    base = Path(filename)
                    files = {}
                    for sp in mixture_fractions:
                        cand = base / f"{sp}.json"
                        if not cand.exists():
                            cand = base / f"{sp}.h5"
                        files[sp] = cand
                else:
                    files = {sp: Path(filename[sp]) for sp in mixture_fractions}
                for i, (sp, path) in enumerate(files.items()):
                    try:
                        rho, T, p, e = load(path)
                    except (FileNotFoundError, KeyError):
                        raise ValueError(f"Missing EOS data for species {sp}") from None
                    w = mixture_fractions[sp]
                    if i == 0:
                        self.rho_grid, self.T_grid = rho, T
                        self.p_val = w * p[0, 0]
                        self.e_val = w * e[0, 0]
                    else:
                        self.p_val += w * p[0, 0]
                        self.e_val += w * e[0, 0]
            else:
                rho, T, p, e = load(filename)
                self.rho_grid, self.T_grid = rho, T
                self.p_val = p[0, 0]
                self.e_val = e[0, 0]
            self.p_interp = lambda pts: np.full(len(pts), self.p_val)
            self.e_interp = lambda pts: np.full(len(pts), self.e_val)

__all__ = ["EOSBase", "IdealGasEOS", "TabulatedEOS", "RealGasEOS", "create_eos"]


class EOSBase(Protocol):
    """Common EOS interface."""

    def pressure(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return pressure for density ``rho`` and temperature ``T``."""

    def energy(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return specific internal energy for density ``rho`` and temperature ``T``."""

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Return temperature for density ``rho`` and specific internal energy ``e``."""


@dataclass
class IdealGasEOS:
    """Ideal gas with constant heat capacity."""

    gamma: float = 5.0 / 3.0

    def pressure(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        return (self.gamma - 1.0) * rho * self.energy(rho, T)

    def energy(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:  # noqa: ARG002
        cv = 1.0 / (self.gamma - 1.0)
        return cv * T

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:  # noqa: ARG002
        cv = 1.0 / (self.gamma - 1.0)
        return e / cv


class TabulatedEOS:
    """Equation of state based on tabulated density/temperature data."""

    def __init__(
        self,
        filename: str | Path | dict[str, str | Path],
        mixture_fractions: dict[str, float] | str | None = None,
    ) -> None:
        """Load EOS tables from ``filename``.

        The underlying implementation reuses
        :class:`dpf2.simulation.eos.TabulatedEOS` to avoid code
        duplication.  The table must contain ``rho`` and ``T`` axes and
        datasets ``p`` (pressure) and ``e`` (specific internal energy).
        """

        if mixture_fractions is not None:
            if isinstance(mixture_fractions, str):
                parts = [p.split(":") for p in mixture_fractions.split(",") if p]
                mixture_fractions = {sp: float(frac) for sp, frac in parts}
            if any(v < 0.0 for v in mixture_fractions.values()):
                raise ValueError("Mixture fractions must be non-negative")
            total = sum(mixture_fractions.values())
            if not np.isclose(total, 1.0):
                raise ValueError("Mixture fractions must sum to one")
            # Ensure all required species tables exist before loading
            if isinstance(filename, (str, Path)):
                base = Path(filename)
                for sp in mixture_fractions:
                    if not (base / f"{sp}.json").exists() and not (base / f"{sp}.h5").exists():
                        raise ValueError(f"Missing EOS data for species {sp}")
            else:
                for sp, sp_path in filename.items():
                    if not Path(sp_path).exists():
                        raise ValueError(f"Missing EOS data for species {sp}")

        self._impl = _SimulationTabulatedEOS(
            filename, mixture_fractions=mixture_fractions
        )
        # Convenience references used for interpolation and inversion
        self.rho_grid = self._impl.rho_grid
        self.T_grid = self._impl.T_grid
        self.p_interp: RegularGridInterpolator = self._impl.p_interp
        self.e_interp: RegularGridInterpolator = self._impl.e_interp

    def pressure(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Interpolate pressure for density ``rho`` and temperature ``T``."""

        points = np.stack([rho, T], axis=-1)
        return self.p_interp(points)

    def energy(self, rho: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Interpolate energy for density ``rho`` and temperature ``T``."""

        points = np.stack([rho, T], axis=-1)
        return self.e_interp(points)

    def temperature(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Invert the energy table to obtain temperature."""

        def _solve(rho_val: float, e_val: float) -> float:
            def func(T):
                return self.energy(np.array([rho_val]), np.array([T]))[0] - e_val

            result = root_scalar(func, bracket=[self.T_grid[0], self.T_grid[-1]])
            return result.root

        return np.vectorize(_solve)(rho, e)


class RealGasEOS(TabulatedEOS):
    """Multi‑species real gas EOS using tabulated thermochemistry.

    This lightweight implementation builds upon :class:`TabulatedEOS` and
    simply enforces that a mixture definition is supplied.  Individual
    species tables are combined according to the provided fractions to
    yield an effective pressure and internal energy for the mixture.  The
    ``temperature`` method is inherited from :class:`TabulatedEOS` and
    therefore performs a one–dimensional root find based on the tabulated
    energy.
    """

    def __init__(
        self,
        filename: str | Path | dict[str, str | Path],
        mixture_fractions: dict[str, float] | str,
    ) -> None:
        if mixture_fractions is None:
            raise ValueError("RealGasEOS requires mixture_fractions")

        # ``TabulatedEOS`` accepts either a mapping or a string of the form
        # ``"A:0.5,B:0.5"``.  Normalise to a dictionary to perform basic
        # validation here before delegating to the parent class.
        if isinstance(mixture_fractions, str):
            parts = [p.split(":") for p in mixture_fractions.split(",") if p]
            mixture_fractions = {sp: float(frac) for sp, frac in parts}

        if any(v < 0.0 for v in mixture_fractions.values()):
            raise ValueError("Mixture fractions must be non‑negative")
        total = sum(mixture_fractions.values())
        if not np.isclose(total, 1.0):
            raise ValueError("Mixture fractions must sum to one")

        super().__init__(filename=filename, mixture_fractions=mixture_fractions)


def create_eos(
    model: EOSModel,
    *,
    table_path: Path | None = None,
    gamma: float = 5.0 / 3.0,
    mixture_fractions: dict[str, float] | str | None = None,
) -> EOSBase:
    """Factory for EOS implementations."""

    if model is EOSModel.IDEAL:
        return IdealGasEOS(gamma=gamma)
    if model is EOSModel.TABULATED and table_path is not None:
        return TabulatedEOS(Path(table_path), mixture_fractions=mixture_fractions)
    if model is EOSModel.REAL_GAS and table_path is not None:
        if mixture_fractions is None:
            raise ValueError("RealGasEOS requires mixture_fractions")
        return RealGasEOS(Path(table_path), mixture_fractions=mixture_fractions)
    raise ValueError(f"Unsupported EOS model: {model}")

