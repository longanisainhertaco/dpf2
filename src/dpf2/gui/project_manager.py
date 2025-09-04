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

        results = run_parametric_sweep(base_config, parameter, values, output_dir=output_dir)
        metrics = compute_sweep_metrics(base_config, results)
        self.metrics[label] = metrics
        return metrics

    def overlay_yield_pressure(self, path: str | Path) -> Path:
        """Generate a yield-versus-pressure overlay plot for stored metrics."""

        return plot_yield_pressure_overlay(self.metrics, path)

    def export_metrics(self, path: str | Path) -> Path:
        """Export all stored metrics to a CSV file.

        Parameters
        ----------
        path:
            Destination path for the CSV file.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "parameter", "yield", "efficiency"])
            for label, metrics in self.metrics.items():
                for param_val, vals in metrics.items():
                    writer.writerow([label, param_val, vals.get("yield", 0.0), vals.get("efficiency", 0.0)])
        return path


__all__ = ["ProjectManager"]
