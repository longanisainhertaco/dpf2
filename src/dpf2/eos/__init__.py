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
        """Very small fallback interpolator when SciPy is unavailable."""

        def __init__(self, points, values):
            self.x, self.y = [np.array(p) for p in points]
            self.values = np.array(values)

        def __call__(self, pts):  # noqa: D401 - behave like SciPy callable
            result = []
            for x, y in pts:
                # locate cell indices
                i = 0
                while i < len(self.x) - 2 and x > self.x[i + 1]:
                    i += 1
                j = 0
                while j < len(self.y) - 2 and y > self.y[j + 1]:
                    j += 1
                x0, x1 = self.x[i], self.x[i + 1]
                y0, y1 = self.y[j], self.y[j + 1]
                tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
                ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
                f00 = self.values[i, j]
                f01 = self.values[i, j + 1]
                f10 = self.values[i + 1, j]
                f11 = self.values[i + 1, j + 1]
                f = (
                    f00 * (1 - tx) * (1 - ty)
                    + f01 * (1 - tx) * ty
                    + f10 * tx * (1 - ty)
                    + f11 * tx * ty
                )
                result.append(f)
            return np.array(result)

    def root_scalar(func, bracket):  # type: ignore[misc]
        a, b = bracket
        fa, fb = func(a), func(b)
        for _ in range(50):
            c = 0.5 * (a + b)
            fc = func(c)
            if abs(fc) < 1e-8:
                return type("_Result", (), {"root": c})()
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return type("_Result", (), {"root": c})()

try:
    from ..core_schema import EOSModel
except Exception:  # pragma: no cover - minimal fallback without pydantic
    from enum import Enum

    class EOSModel(str, Enum):
        IDEAL = "ideal"
        TABULATED = "tabulated"
        REAL_GAS = "real_gas"

import json
try:  # pragma: no cover - optional h5py dependency
    import h5py  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    h5py = None  # type: ignore

__all__ = ["EOSBase", "IdealGasEOS", "TabulatedEOS", "RealGasEOS", "create_eos"]


def _parse_mixture_fractions(
    mixture_fractions: dict[str, float] | str | None,
) -> dict[str, float] | None:
    """Normalise ``mixture_fractions`` input.

    ``mixture_fractions`` may be provided either as a mapping or as a string
    of the form ``"Ar:0.9,H:0.1"``.  ``None`` is returned unchanged which is
    convenient for single species tables.  The helper lives at module scope so
    that both :class:`TabulatedEOS` and :class:`RealGasEOS` can make use of the
    same parsing logic.
    """

    if mixture_fractions is None:
        return None
    if isinstance(mixture_fractions, str):
        parts = [p.split(":") for p in mixture_fractions.split(",") if p]
        mixture_fractions = {sp: float(frac) for sp, frac in parts}
    return mixture_fractions


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
    """Equation of state based on tabulated density/temperature data.

    The table may represent either a single species or a mixture of
    multiple species specified via ``mixture_fractions``.  Data can be
    supplied in JSON or HDF5 format and must provide ``rho`` and ``T``
    axes with corresponding pressure ``p`` and specific internal energy
    ``e`` values on the grid.
    """

    def __init__(
        self,
        filename: str | Path | dict[str, str | Path],
        mixture_fractions: dict[str, float] | str | None = None,
    ) -> None:
        mixture_fractions = _parse_mixture_fractions(mixture_fractions)

        if mixture_fractions is None:
            if isinstance(filename, dict):
                raise ValueError("Mapping of files only valid for mixtures")
            rho, T, p, e = self._load_table(Path(filename))
        else:
            if any(v < 0.0 for v in mixture_fractions.values()):
                raise ValueError("Mixture fractions must be non-negative")
            total = sum(mixture_fractions.values())
            if not np.isclose(total, 1.0):
                raise ValueError("Mixture fractions must sum to one")

            if isinstance(filename, (str, Path)):
                base = Path(filename)
                files: dict[str, Path] = {}
                for sp in mixture_fractions:
                    cand = base / f"{sp}.json"
                    if not cand.exists():
                        cand = base / f"{sp}.h5"
                    if not cand.exists():
                        raise ValueError(f"Missing EOS data for species {sp}")
                    files[sp] = cand
            else:
                files = {sp: Path(pth) for sp, pth in filename.items()}
                for sp, sp_path in files.items():
                    if not sp_path.exists():
                        raise ValueError(f"Missing EOS data for species {sp}")

            for i, (sp, path) in enumerate(files.items()):
                rho_i, T_i, p_i, e_i = self._load_table(path)
                w = mixture_fractions[sp]
                if i == 0:
                    rho, T = rho_i, T_i
                    p = w * p_i
                    e = w * e_i
                else:
                    if not (np.allclose(rho, rho_i) and np.allclose(T, T_i)):
                        raise ValueError("Species tables must share the same rho/T grid")
                    p += w * p_i
                    e += w * e_i

        self.rho_grid = rho
        self.T_grid = T
        self.p_interp: RegularGridInterpolator = RegularGridInterpolator((rho, T), p)
        self.e_interp: RegularGridInterpolator = RegularGridInterpolator((rho, T), e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_table(path: Path):
        """Return ``rho``, ``T``, ``p`` and ``e`` arrays from ``path``."""

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
        if path.suffix in {".h5", ".hdf5"}:
            if h5py is None:  # pragma: no cover - exercised in environments without h5py
                raise ModuleNotFoundError("h5py is required for tabulated EOS")
            with h5py.File(path, "r") as f:  # type: ignore[assignment]
                return (
                    np.array(f["rho"][:]),
                    np.array(f["T"][:]),
                    np.array(f["p"][:]),
                    np.array(f["e"][:]),
                )
        raise ValueError(f"Unsupported EOS file format: {path}")

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
        mixture_fractions = _parse_mixture_fractions(mixture_fractions)
        if mixture_fractions is None:
            raise ValueError("RealGasEOS requires mixture_fractions")

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

