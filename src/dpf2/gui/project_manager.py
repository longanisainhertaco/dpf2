from __future__ import annotations

"""Simple project management utilities for parametric studies.

This module provides a lightweight :class:`ProjectManager` class that wraps the
existing optimization helpers in :mod:`dpf2.optimization.param_sweep` to ease
running sweeps, comparing results and exporting metrics.  The class stores
metrics from multiple sweeps which can then be overlaid or written to disk.
"""

from pathlib import Path
import csv
from typing import Dict, Iterable

from ..core.config import DPFConfig
from ..optimization.param_sweep import (
    run_parametric_sweep,
    compute_sweep_metrics,
    plot_yield_pressure_overlay,
)


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
        metrics = compute_sweep_metrics(base_config, results)
        self.metrics[label] = metrics
        self.params[label] = parameter
        return metrics

    def overlay_yield_pressure(self, path: str | Path) -> Path:
        """Generate a yield-versus-pressure overlay plot for stored metrics."""

        return plot_yield_pressure_overlay(self.metrics, path)

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
