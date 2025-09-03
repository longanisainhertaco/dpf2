"""Utilities for performing parametric sweeps of simulation parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..core.config import DPFConfig
from ..core.simulation import DPFSimulation


SweepResult = Tuple[List[float], List[float], List[float]]


def run_parametric_sweep(
    base_config: DPFConfig,
    parameter: str,
    values: Iterable[float],
    *,
    output_dir: str | Path = "sweep_output",
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
        t, i, v = sim.run(output_dir=str(run_dir))
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
    base_config: DPFConfig, results: Dict[float, SweepResult]
) -> Dict[float, Dict[str, float]]:
    """Compute simple yield and efficiency estimates for sweep results.

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
        Mapping of parameter value to metrics ``{"yield", "efficiency"}``.
        Yield is estimated as the peak current, while efficiency is the ratio
        of the time-integrated ``I*V`` product to the initial stored energy.
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
        yield_est = float(i_arr.max())
        metrics[val] = {"yield": yield_est, "efficiency": efficiency}

    return metrics


def plot_metric_overlay(
    parameter: str,
    metrics: Dict[float, Dict[str, float]],
    path: str | Path,
) -> Path:
    """Plot yield and efficiency against the swept parameter on shared axes."""

    import matplotlib.pyplot as plt

    vals = sorted(metrics.keys())
    yields = [metrics[v]["yield"] for v in vals]
    effs = [metrics[v]["efficiency"] for v in vals]

    fig, ax1 = plt.subplots()
    color1 = "tab:blue"
    ax1.set_xlabel(parameter)
    ax1.set_ylabel("Yield", color=color1)
    ax1.plot(vals, yields, color=color1, marker="o", label="yield")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Efficiency", color=color2)
    ax2.plot(vals, effs, color=color2, marker="s", label="efficiency")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = [
    "run_parametric_sweep",
    "plot_sweep_results",
    "compute_sweep_metrics",
    "plot_metric_overlay",
    "SweepResult",
]
