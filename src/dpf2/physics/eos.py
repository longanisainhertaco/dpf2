from __future__ import annotations

"""Helpers for tabulated equations of state and opacity data."""

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import List, Sequence
import json


@dataclass
class TabulatedEOS:
    """Tabulated EOS with optional opacity data."""

    rho: Sequence[float]
    temperature: Sequence[float]
    pressure: Sequence[Sequence[float]]
    energy: Sequence[Sequence[float]]
    opacity: Sequence[Sequence[float]] | None = None

    def _interp1d(self, x: float, xp: Sequence[float], fp: Sequence[float]) -> float:
        if x <= xp[0]:
            return fp[0]
        if x >= xp[-1]:
            return fp[-1]
        for i in range(len(xp) - 1):
            if xp[i] <= x <= xp[i + 1]:
                t = (x - xp[i]) / (xp[i + 1] - xp[i]) if xp[i + 1] != xp[i] else 0.0
                return fp[i] * (1 - t) + fp[i + 1] * t
        return fp[-1]

    def _interp2d(self, x: float, y: float, table: Sequence[Sequence[float]]) -> float:
        rows = [self._interp1d(y, self.temperature, row) for row in table]
        return self._interp1d(x, self.rho, rows)

    def pressure_at(self, rho: float, T: float) -> float:
        return self._interp2d(rho, T, self.pressure)

    def energy_at(self, rho: float, T: float) -> float:
        return self._interp2d(rho, T, self.energy)

    def opacity_at(self, rho: float, T: float) -> float:
        if self.opacity is None:
            raise ValueError("opacity table not available")
        return self._interp2d(rho, T, self.opacity)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_tabulated_eos(path: str | Path) -> TabulatedEOS:
    """Load EOS/opacity data from ``path``."""

    with open(path) as f:
        data = json.load(f)
    def _to_float_list(seq):
        return [float(x) for x in seq]

    def _to_2d_list(seq2d):
        return [[float(x) for x in row] for row in seq2d]

    return TabulatedEOS(
        rho=_to_float_list(data["rho"]),
        temperature=_to_float_list(data["T"]),
        pressure=_to_2d_list(data["p"]),
        energy=_to_2d_list(data["e"]),
        opacity=_to_2d_list(data["opacity"]) if "opacity" in data else None,
    )


def load_standard_eos(name: str) -> TabulatedEOS:
    """Load one of the small built-in EOS tables distributed with the package."""

    file = resources.files("dpf2.eos") / f"{name}.json"
    with resources.as_file(file) as path:
        return load_tabulated_eos(path)


__all__ = ["TabulatedEOS", "load_tabulated_eos", "load_standard_eos"]
