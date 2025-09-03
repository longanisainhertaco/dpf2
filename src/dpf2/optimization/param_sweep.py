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


__all__ = ["run_parametric_sweep", "plot_sweep_results", "SweepResult"]
