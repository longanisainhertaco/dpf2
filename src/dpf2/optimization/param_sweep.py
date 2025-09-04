"""Utilities for performing parametric sweeps of simulation parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dataclasses import asdict

from ..core.config import DPFConfig
from ..core.simulation import DPFSimulation
from dpf2.cli.lab import write_manifest
import random
import numpy as np


SweepResult = Tuple[List[float], List[float], List[float]]


def run_parametric_sweep(
    base_config: DPFConfig,
    parameter: str,
    values: Iterable[float],
    *,
    output_dir: str | Path = "sweep_output",
    lab_mode: bool = False,
    config_path: str | Path | None = None,
) -> Dict[float, SweepResult]:
    """Run a series of simulations while varying a single parameter.

    Parameters
    ----------
    base_config:
        Starting :class:`DPFConfig` for the sweep.
    parameter:
        Name of the configuration attribute to vary.
    values:
        Iterable of values for ``parameter``.
    output_dir:
        Directory where per-run results are written.

    Returns
    -------
    Dict[float, SweepResult]
        Mapping of parameter value to time history tuples ``(t, I, V)``.
    """

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[float, SweepResult] = {}
    for val in values:
        cfg_dict = base_config.__dict__.copy()
        cfg_dict[parameter] = val
        cfg = DPFConfig(**cfg_dict)
        sim = DPFSimulation(cfg)
        run_dir = out_root / f"{parameter}_{val}"
        if lab_mode:
            seeds = {"python": random.getstate()[1][0]}
            try:
                seeds["numpy"] = int(np.random.get_state()[1][0])
            except Exception:
                try:
                    rng = np.random.default_rng()
                    seeds["numpy"] = int(rng.bit_generator.state["state"]["state"])
                except Exception:
                    seeds["numpy"] = 0
        t, i, v = sim.run(output_dir=str(run_dir))
        if lab_mode:
            ppc = getattr(getattr(cfg, "warpx_settings", None), "max_particles_per_cell", None)
            paths = [str(config_path)] if config_path else []
            write_manifest(
                run_dir,
                config_paths=paths,
                config=asdict(cfg),
                ppc=ppc,
                seeds=seeds,
            )
        results[val] = (t, i, v)
    return results


def plot_sweep_results(
    parameter: str,
    results: Dict[float, SweepResult],
    path: str | Path,
) -> Path:
    """Create an overlay plot of currents for a sweep."""

    import matplotlib.pyplot as plt

    for val, (t, current, _v) in results.items():
        plt.plot(t, current, label=f"{parameter}={val:g}")

    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")
    plt.legend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    return path


def compute_sweep_metrics(
    base_config: DPFConfig,
    results: Dict[float, SweepResult],
    parameter: str | None = None,
) -> Dict[float, Dict[str, float]]:
    """Compute simple yield, pinch time and efficiency estimates for sweep results.

    Parameters
    ----------
    base_config:
        Configuration used for the sweep.  The capacitance and charging
        voltage are used to estimate the initial energy in the circuit.
    results:
        Output of :func:`run_parametric_sweep`.

    Returns
    -------
    Dict[float, Dict[str, float]]
        Mapping of parameter value to metrics ``{"yield", "pinch_time",
        "efficiency"}``.  Yield is estimated as the peak current, pinch time is
        the time at which the peak current occurs and efficiency is the ratio of
        the time-integrated ``I*V`` product to the initial stored energy.  If
        ``parameter`` is provided and equals ``"initial_pressure"``, the
        dimensionless shock parameter ``S = I/(a*p0)`` is also recorded for each
        sweep value.
    """

    import numpy as np

    metrics: Dict[float, Dict[str, float]] = {}
    energy_in = 0.5 * base_config.capacitance * base_config.charging_voltage**2

    for val, (t, current, voltage) in results.items():
        t_arr = np.array(t)
        i_arr = np.array(current)
        v_arr = np.array(voltage)
        power = i_arr * v_arr
        energy_out = float(np.trapz(power, t_arr))
        efficiency = energy_out / energy_in if energy_in else 0.0
        peak_idx = int(i_arr.argmax()) if len(i_arr) else 0
        yield_est = float(i_arr[peak_idx])
        pinch_time = float(t_arr[peak_idx]) if len(t_arr) else 0.0
        metric = {
            "yield": yield_est,
            "efficiency": efficiency,
            "pinch_time": pinch_time,
        }
        if parameter == "initial_pressure":
            pressure = val
        else:
            pressure = base_config.initial_pressure
        a = getattr(base_config, "anode_radius", 0.0)
        if a > 0 and pressure > 0:
            metric["S"] = yield_est / (a * pressure)
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
    effs = [metrics[v]["efficiency"] for v in vals]

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
    "plot_sweep_results",
    "compute_sweep_metrics",
    "plot_metric_overlay",
    "plot_yield_vs_S",
    "plot_yield_pressure_overlay",
    "SweepResult",
]
