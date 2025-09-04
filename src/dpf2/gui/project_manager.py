from __future__ import annotations

"""Simple project management utilities for parametric studies.

This module provides a lightweight :class:`ProjectManager` class that wraps the
existing optimization helpers in :mod:`dpf2.optimization.param_sweep` to ease
running sweeps, comparing results and exporting metrics.  The class stores
metrics from multiple sweeps which can then be overlaid or written to disk.
"""

from pathlib import Path
import csv
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..core.config import DPFConfig
from ..optimization.param_sweep import (
    run_parametric_sweep,
    compute_sweep_metrics,
    plot_yield_pressure_overlay,
    plot_yield_vs_S as _plot_yield_vs_S,
)
from ..optimization.multi_objective import random_pareto_search


class ProjectManager:
    """Manage simulation sweeps and KPI comparisons.

    Examples
    --------
    >>> cfg = DPFConfig()
    >>> pm = ProjectManager()
    >>> pm.run_sweep("baseline", cfg, "initial_pressure", [0.5, 1.0])  # doctest: +SKIP
    >>> pm.export_metrics("metrics.csv")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, Dict[float, Dict[str, float]]] = {}
        self.params: Dict[str, str] = {}

    @staticmethod
    def _spot_size(t: Iterable[float], current: Iterable[float]) -> float:
        """Estimate spot size as the FWHM of the current trace."""

        t_arr = np.array(list(t))
        i_arr = np.array(list(current))
        if len(t_arr) == 0:
            return 0.0
        peak = float(i_arr.max())
        half = 0.5 * peak
        mask = i_arr >= half
        if not mask.any():
            return 0.0
        return float(t_arr[mask][-1] - t_arr[mask][0])

    def run_sweep(
        self,
        label: str,
        base_config: DPFConfig,
        parameter: str,
        values: Iterable[float],
        *,
        output_dir: str | Path = "sweep_output",
    ) -> Dict[float, Dict[str, float]]:
        """Run a parametric sweep and store computed metrics.

        Parameters
        ----------
        label:
            Identifier for the sweep results.
        base_config:
            Base configuration to mutate for each sweep value.
        parameter:
            Name of :class:`~dpf2.core.config.DPFConfig` attribute to vary.
        values:
            Iterable of values for ``parameter``.
        output_dir:
            Directory where individual run results should be written.
        """

        results = run_parametric_sweep(
            base_config, parameter, values, output_dir=output_dir
        )
        metrics = compute_sweep_metrics(base_config, results, parameter)
        self.metrics[label] = metrics
        self.params[label] = parameter
        return metrics

    def overlay_yield_pressure(self, path: str | Path) -> Path:
        """Generate a yield-versus-pressure overlay plot for stored metrics."""

        return plot_yield_pressure_overlay(self.metrics, path)

    def plot_yield_vs_S(self, label: str, path: str | Path) -> Path:
        """Plot yield versus shock parameter ``S`` for a stored sweep."""

        metrics = self.metrics.get(label, {})
        return _plot_yield_vs_S(metrics, path)

    def pareto_search(
        self,
        base_config: DPFConfig,
        bounds: Dict[str, Tuple[float, float]],
        n_samples: int = 100,
    ) -> List[Dict[str, float]]:
        """Run a random Pareto search for yield versus spot size."""

        eval_cache: Dict[Tuple[float, ...], Tuple[float, float]] = {}

        names = list(bounds)

        def evaluate(params: np.ndarray) -> Tuple[float, float]:
            cfg_dict = base_config.__dict__.copy()
            for idx, name in enumerate(names):
                cfg_dict[name] = float(params[idx])
            cfg = DPFConfig(**cfg_dict)
            from ..core.simulation import DPFSimulation

            sim = DPFSimulation(cfg)
            t, i, v = sim.run()
            yield_val = float(max(i)) if len(i) else 0.0
            spot = self._spot_size(t, i)
            eval_cache[tuple(params)] = (yield_val, spot)
            return yield_val, spot

        pareto_params = random_pareto_search(evaluate, bounds, n_samples=n_samples)
        front: List[Dict[str, float]] = []
        for param_dict in pareto_params:
            params = tuple(param_dict[name] for name in names)
            y, s = eval_cache[params]
            front.append({**param_dict, "yield": y, "spot_size": s})
        return front

    def overlay_metrics(
        self, path: str | Path, parameter: str | None = None
    ) -> Path:
        """Overlay yield, pinch time and efficiency curves for stored sweeps.

        Parameters
        ----------
        path:
            Destination image path.
        parameter:
            Optional axis label. If omitted and all recorded sweeps share a
            common parameter, that name is used automatically.
        """

        import matplotlib.pyplot as plt

        if parameter is None:
            params = {self.params.get(k, "") for k in self.metrics}
            parameter = params.pop() if len(params) == 1 else "parameter"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, sharex=False, figsize=(12, 4))

        for label, metrics in self.metrics.items():
            vals = sorted(metrics.keys())
            y = [metrics[v].get("yield", 0.0) for v in vals]
            p = [metrics[v].get("pinch_time", 0.0) for v in vals]
            e = [metrics[v].get("efficiency", 0.0) for v in vals]
            axes[0].plot(vals, y, label=label)
            axes[1].plot(vals, p, label=label)
            axes[2].plot(vals, e, label=label)

        axes[0].set_ylabel("Yield")
        axes[1].set_ylabel("Pinch Time")
        axes[2].set_ylabel("Efficiency")
        for ax in axes:
            ax.set_xlabel(parameter)
            ax.grid(True)
            ax.legend()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def export_metrics(self, path: str | Path) -> Path:
        """Export all stored metrics to a CSV file.

        The resulting table contains the sweep label, parameter value and the
        computed ``yield``, ``pinch_time`` and ``efficiency`` metrics.

        Parameters
        ----------
        path:
            Destination path for the CSV file.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "parameter", "yield", "pinch_time", "efficiency"])
            for label, metrics in self.metrics.items():
                for param_val, vals in metrics.items():
                    writer.writerow([
                        label,
                        param_val,
                        vals.get("yield", 0.0),
                        vals.get("pinch_time", 0.0),
                        vals.get("efficiency", 0.0),
                    ])
        return path


__all__ = ["ProjectManager"]
