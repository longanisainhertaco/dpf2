"""Utilities for evaluating surrogate models over parameter sweeps.

This module previously executed the full :class:`~dpf2.core.simulation.DPFSimulation`
for each sweep point and recorded the resulting current traces.  The new
implementation instead queries lightweight surrogate models that predict the
neutron yield and pinch time directly.  Each prediction is accompanied by a
simple conformal uncertainty band.

When a sweep value lies outside the training domain of a surrogate model the
model may raise :class:`OutOfDomainError`.  Such points are skipped and a
warning is emitted so that callers can decide how to handle missing results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple
import warnings

from ..core.config import DPFConfig
from . import OptimizationWarning


class OutOfDomainError(Exception):
    """Raised when querying a surrogate outside its trained domain."""


# A prediction consists of a value and its (lo, hi) conformal band
Prediction = Tuple[float, Tuple[float, float]]

# Each sweep result stores the yield and pinch time predictions
SweepResult = Dict[str, Prediction]


def run_parametric_sweep(
    base_config: DPFConfig,
    parameter: str,
    values: Iterable[float],
    *,
    yield_model: str | Path | None = None,
    pinch_model: str | Path | None = None,
    output_dir: str | Path = "sweep_output",
    lab_mode: bool = False,  # retained for API compatibility
    config_path: str | Path | None = None,
) -> Dict[float, SweepResult]:
    """Evaluate surrogate models for a set of parameter values.

    Parameters
    ----------
    base_config:
        Starting configuration for the sweep.  Currently only used when
        computing derived metrics such as the shock parameter ``S``.
    parameter:
        Name of the configuration attribute to vary.  The value is passed
        directly to the surrogate models.
    values:
        Iterable of values for ``parameter``.
    yield_model, pinch_model:
        Optional paths to surrogate model JSON files.  When omitted the
        built-in repository models are used.
    output_dir, lab_mode, config_path:
        Accepted for backward compatibility but currently unused.  The output
        directory is still created so callers relying on its existence do not
        break.

    Returns
    -------
    Dict[float, SweepResult]
        Mapping of parameter values to prediction dictionaries.  Each
        dictionary contains entries ``{"yield": (y, (lo, hi)), "pinch_time":
        (p, (lo, hi))}``.
    """

    model_dir = Path(__file__).resolve().parents[2] / "ai" / "surrogates"
    try:  # pragma: no cover - optional dependency
        from ..ai.surrogate import ONNXSurrogateModel, OutOfDomainError as _OOD

        y_path = Path(yield_model) if yield_model else model_dir / "yield_model.onnx"
        p_path = Path(pinch_model) if pinch_model else model_dir / "pinch_time_model.onnx"
        y_model = ONNXSurrogateModel.load(y_path)
        p_model = ONNXSurrogateModel.load(p_path)
        OutOfDomainError = _OOD
    except Exception:  # pragma: no cover - fallback to simple models
        from ..ai.simple_surrogates import (
            LinearSurrogate,
            load_pinch_time_surrogate,
            load_yield_surrogate,
            OutOfDomainError as _OOD,
        )

        if yield_model:
            y_model = LinearSurrogate.load(Path(yield_model))
        else:
            y_model = load_yield_surrogate()

        if pinch_model:
            p_model = LinearSurrogate.load(Path(pinch_model))
        else:
            p_model = load_pinch_time_surrogate()
        OutOfDomainError = _OOD

    # Ensure the output directory exists for compatibility with older APIs
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: Dict[float, SweepResult] = {}
    for val in values:
        try:
            y_pred, y_band = y_model.predict_with_uncertainty(val)
            p_pred, p_band = p_model.predict_with_uncertainty(val)
        except OutOfDomainError as exc:
            warnings.warn(str(exc), OptimizationWarning, stacklevel=2)
            continue

        results[float(val)] = {
            "yield": (float(y_pred), (float(y_band[0]), float(y_band[1]))),
            "pinch_time": (float(p_pred), (float(p_band[0]), float(p_band[1]))),
        }

    return results


def compute_sweep_metrics(
    base_config: DPFConfig,
    results: Dict[float, SweepResult],
    parameter: str | None = None,
) -> Dict[float, Dict[str, float]]:
    """Compute simple metrics for surrogate sweep results.

    The returned mapping contains the surrogate predictions along with their
    uncertainty bands.  Efficiency is currently undefined for surrogate-only
    sweeps and is therefore reported as ``0.0``.
    """

    metrics: Dict[float, Dict[str, float]] = {}
    a = getattr(base_config, "anode_radius", 0.0)

    for val, preds in results.items():
        y_pred, y_band = preds.get("yield", (0.0, (0.0, 0.0)))
        p_pred, p_band = preds.get("pinch_time", (0.0, (0.0, 0.0)))
        metric: Dict[str, float] = {
            "yield": float(y_pred),
            "pinch_time": float(p_pred),
            "yield_lo": float(y_band[0]),
            "yield_hi": float(y_band[1]),
            "pinch_time_lo": float(p_band[0]),
            "pinch_time_hi": float(p_band[1]),
            "efficiency": 0.0,
        }

        pressure = val if parameter == "initial_pressure" else base_config.initial_pressure
        if a > 0 and pressure > 0:
            metric["S"] = float(y_pred) / (a * pressure)

        metrics[val] = metric

    return metrics


def plot_metric_overlay(
    parameter: str,
    metrics: Dict[float, Dict[str, float]],
    path: str | Path,
) -> Path:
    """Plot yield, pinch time and efficiency against a swept parameter."""

    import matplotlib.pyplot as plt

    vals = sorted(metrics.keys())
    yields = [metrics[v]["yield"] for v in vals]
    pinch = [metrics[v].get("pinch_time", 0.0) for v in vals]
    effs = [metrics[v].get("efficiency", 0.0) for v in vals]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 9))

    axes[0].plot(vals, yields, marker="o")
    axes[0].set_ylabel("Yield")

    axes[1].plot(vals, pinch, marker="^")
    axes[1].set_ylabel("Pinch Time")

    axes[2].plot(vals, effs, marker="s")
    axes[2].set_ylabel("Efficiency")
    axes[2].set_xlabel(parameter)

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_yield_vs_S(metrics: Dict[float, Dict[str, float]], path: str | Path) -> Path:
    """Plot yield as a function of the shock parameter ``S``."""

    import matplotlib.pyplot as plt

    pairs = sorted((m.get("S", 0.0), m.get("yield", 0.0)) for m in metrics.values())
    s_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    plt.figure()
    plt.plot(s_vals, y_vals, marker="o")
    plt.xlabel("S")
    plt.ylabel("Yield")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    return path


def plot_yield_pressure_overlay(
    metric_sets: Dict[str, Dict[float, Dict[str, float]]],
    path: str | Path,
) -> Path:
    """Overlay yield vs. pressure curves for multiple sweeps."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for label, metrics in metric_sets.items():
        pressures = sorted(metrics.keys())
        yields = [metrics[p]["yield"] for p in pressures]
        plt.plot(pressures, yields, marker="o", label=label)
    plt.xlabel("Pressure")
    plt.ylabel("Yield")
    plt.legend()
    plt.savefig(path)
    plt.close()
    return path


__all__ = [
    "run_parametric_sweep",
    "compute_sweep_metrics",
    "plot_metric_overlay",
    "plot_yield_vs_S",
    "plot_yield_pressure_overlay",
    "SweepResult",
    "Prediction",
]

