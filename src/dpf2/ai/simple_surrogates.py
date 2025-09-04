"""Lightweight runtime helpers for linear surrogate models.

The surrogates are trained offline and stored as JSON files containing the
linear coefficients, the training domain and an estimate of the training
error.  At runtime the helpers load these files and perform predictions while
enforcing out-of-distribution checks to prevent evaluations outside the
training hull.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .surrogate import ONNXSurrogateModel
from ..exceptions import OutOfDomainError


@dataclass
class LinearSurrogate:
    """Simple linear regression surrogate ``y = a*x + b``."""

    coeffs: Sequence[float]
    domain: Sequence[float]
    error: float
    mean: float | None = None
    covariance: float | None = None
    ood_threshold: float | None = None

    def predict(self, x: float | Iterable[float]) -> float | list[float]:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            inputs = list(x)
            return [self._predict_single(val) for val in inputs]
        return self._predict_single(float(x))

    def predict_with_uncertainty(
        self, x: float | Iterable[float]
    ) -> tuple[float, tuple[float, float]] | list[tuple[float, tuple[float, float]]]:
        """Return prediction and 95% conformal interval."""

        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            inputs = list(x)
            return [self._predict_with_uncertainty_single(val) for val in inputs]
        return self._predict_with_uncertainty_single(float(x))

    def _predict_single(self, val: float) -> float:
        lo, hi = self.domain
        dist = self._mahalanobis(val)
        if val < lo or val > hi or (
            self.ood_threshold is not None and dist > self.ood_threshold
        ):
            raise OutOfDomainError(
                f"Input {val} outside training range [{lo}, {hi}] (distance={dist:.2f})"
            )
        a, b = self.coeffs
        return a * val + b

    def _predict_with_uncertainty_single(
        self, val: float
    ) -> tuple[float, tuple[float, float]]:
        pred = self._predict_single(val)
        dist = self._mahalanobis(val)
        band = self.error * (1.0 + dist)
        return pred, (pred - band, pred + band)

    def _mahalanobis(self, val: float) -> float:
        if self.mean is None or self.covariance is None:
            return 0.0
        diff = val - self.mean
        return float(abs(diff) / np.sqrt(self.covariance))

    @classmethod
    def load(cls, path: Path) -> "LinearSurrogate":
        with path.open() as fh:
            data = json.load(fh)
        coeffs = data.get("coeffs", [0.0, 0.0])
        domain = data.get("training_domain", [0.0, 0.0])
        error = data.get("error", 0.0)
        mean = data.get("mean")
        covariance = data.get("covariance")
        threshold = data.get("ood_threshold")
        return cls(
            coeffs=coeffs,
            domain=domain,
            error=error,
            mean=mean,
            covariance=covariance,
            ood_threshold=threshold,
        )


@dataclass
class ONNXLinearSurrogate:
    """Wrapper around :class:`ONNXSurrogateModel` with domain checks."""

    model: ONNXSurrogateModel
    domain: Sequence[float]
    error: float
    mean: float | None = None
    covariance: float | None = None
    ood_threshold: float | None = None

    def predict(self, x: float | Iterable[float]) -> float | list[float]:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            arr = np.asarray(list(x), dtype=np.float32).reshape(-1, 1)
            return [self._predict_single(v) for v in arr[:, 0]]
        return float(self._predict_single(float(x)))

    def predict_with_uncertainty(
        self, x: float | Iterable[float]
    ) -> tuple[float, tuple[float, float]] | list[tuple[float, tuple[float, float]]]:
        """Return prediction and 95% conformal interval."""
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            arr = np.asarray(list(x), dtype=np.float32).reshape(-1, 1)
            return [self._predict_with_uncertainty_single(v) for v in arr[:, 0]]
        return self._predict_with_uncertainty_single(float(x))

    def _predict_single(self, val: float) -> float:
        lo, hi = self.domain
        dist = self._mahalanobis(val)
        if val < lo or val > hi or (
            self.ood_threshold is not None and dist > self.ood_threshold
        ):
            raise OutOfDomainError(
                f"Input {val} outside training range [{lo}, {hi}] (distance={dist:.2f})"
            )
        inp = np.array([[val]], dtype=np.float32)
        return float(self.model.predict(inp)[0, 0])

    def _predict_with_uncertainty_single(
        self, val: float
    ) -> tuple[float, tuple[float, float]]:
        pred = self._predict_single(val)
        dist = self._mahalanobis(val)
        band = self.error * (1.0 + dist)
        return pred, (pred - band, pred + band)

    def _mahalanobis(self, val: float) -> float:
        if self.mean is None or self.covariance is None:
            return 0.0
        diff = val - self.mean
        return float(abs(diff) / np.sqrt(self.covariance))


# Convenience loaders ---------------------------------------------------------
# Repository root is three levels up from this file
MODEL_DIR = Path(__file__).resolve().parents[3] / "ai" / "surrogates"


def load_yield_surrogate() -> LinearSurrogate | ONNXLinearSurrogate:
    """Return the surrogate model for neutron yield."""

    meta = MODEL_DIR / "yield_model.json"
    with meta.open() as fh:
        data = json.load(fh)

    domain = data.get("training_domain", [0.0, 0.0])
    error = data.get("error", 0.0)
    coeffs = data.get("coeffs", [0.0, 0.0])
    mean = data.get("mean")
    covariance = data.get("covariance")
    threshold = data.get("ood_threshold")
    onnx_file = data.get("onnx")

    if onnx_file:
        onnx_path = meta.parent / onnx_file
        try:
            model = ONNXSurrogateModel.load(onnx_path)
            return ONNXLinearSurrogate(
                model=model,
                domain=domain,
                error=error,
                mean=mean,
                covariance=covariance,
                ood_threshold=threshold,
            )
        except Exception:
            pass

    return LinearSurrogate(
        coeffs=coeffs,
        domain=domain,
        error=error,
        mean=mean,
        covariance=covariance,
        ood_threshold=threshold,
    )


def load_pinch_time_surrogate() -> LinearSurrogate:
    """Return the surrogate model for pinch time."""

    return LinearSurrogate.load(MODEL_DIR / "pinch_time_model.json")


__all__ = [
    "LinearSurrogate",
    "ONNXLinearSurrogate",
    "load_yield_surrogate",
    "load_pinch_time_surrogate",
]
