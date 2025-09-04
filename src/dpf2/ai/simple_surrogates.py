"""Lightweight runtime helpers for linear surrogate models.

The surrogates are trained offline and stored as JSON files containing the
linear coefficients, the training domain and an estimate of the training
error.  At runtime the helpers load these files and perform predictions while
emitting a warning when inputs fall outside the training range.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class LinearSurrogate:
    """Simple linear regression surrogate ``y = a*x + b``."""

    coeffs: Sequence[float]
    domain: Sequence[float]
    error: float

    def predict(self, x: float | Iterable[float]) -> float | list[float]:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            inputs = list(x)
            return [self._predict_single(val) for val in inputs]
        return self._predict_single(float(x))

    def _predict_single(self, val: float) -> float:
        lo, hi = self.domain
        if val < lo or val > hi:
            warnings.warn(
                f"Input {val} outside training range [{lo}, {hi}]",
                RuntimeWarning,
                stacklevel=2,
            )
        a, b = self.coeffs
        return a * val + b

    @classmethod
    def load(cls, path: Path) -> "LinearSurrogate":
        with path.open() as fh:
            data = json.load(fh)
        coeffs = data.get("coeffs", [0.0, 0.0])
        domain = data.get("training_domain", [0.0, 0.0])
        error = data.get("error", 0.0)
        return cls(coeffs=coeffs, domain=domain, error=error)


# Convenience loaders ---------------------------------------------------------
# Repository root is three levels up from this file
MODEL_DIR = Path(__file__).resolve().parents[3] / "ai" / "surrogates"


def load_yield_surrogate() -> LinearSurrogate:
    """Return the surrogate model for neutron yield."""

    return LinearSurrogate.load(MODEL_DIR / "yield_model.json")


def load_pinch_time_surrogate() -> LinearSurrogate:
    """Return the surrogate model for pinch time."""

    return LinearSurrogate.load(MODEL_DIR / "pinch_time_model.json")


__all__ = [
    "LinearSurrogate",
    "load_yield_surrogate",
    "load_pinch_time_surrogate",
]
