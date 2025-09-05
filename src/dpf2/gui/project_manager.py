from __future__ import annotations

"""Simple project management utilities for parametric studies.

This module provides a lightweight :class:`ProjectManager` class that wraps the
existing optimization helpers in :mod:`dpf2.optimization.param_sweep` to ease
running sweeps, comparing results and exporting metrics.  The class stores
metrics from multiple sweeps which can then be overlaid or written to disk.
"""

from pathlib import Path
import csv
import json
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
try:  # pragma: no cover - optional dependency for circuit visualisation
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from ..core.config import DPFConfig
from ..optimization.param_sweep import (
    run_parametric_sweep,
    compute_sweep_metrics,
    plot_yield_pressure_overlay,
    plot_yield_vs_S as _plot_yield_vs_S,
)
from ..optimization.multi_objective import (
    ConvergenceRecord,
    nsga2,
    random_pareto_search,
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

    def __init__(self, project: str = "default") -> None:
        self.metrics: Dict[str, Dict[float, Dict[str, float]]] = {}
        self.params: Dict[str, str] = {}
        self.project = project
        self.last_kpi_plot: Path | None = None
        self.last_convergence_plot: Path | None = None
        # Geometry and circuit state
        self.geometries: Dict[str, Any] = {}
        self.circuit: nx.Graph | None = nx.Graph() if nx else None

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

    def _save_kpi_plot(
        self, label: str, parameter: str, metrics: Dict[float, Dict[str, float]]
    ) -> Path:
        """Persist KPI plots for a single sweep.

        A three-panel figure is created showing yield, wall-plug efficiency
        and spot size versus the swept ``parameter``.  The resulting image is
        stored under ``results/<project>/kpi_plots/`` and the path recorded on
        ``last_kpi_plot`` for later display.
        """

        import matplotlib.pyplot as plt

        vals = sorted(metrics.keys())
        y = [metrics[v].get("yield", 0.0) for v in vals]
        eff = [
            metrics[v].get("wall_plug_efficiency", metrics[v].get("efficiency", 0.0))
            for v in vals
        ]
        spot = [metrics[v].get("spot_size", 0.0) for v in vals]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].plot(vals, y, marker="o")
        axes[0].set_ylabel("Yield")
        axes[1].plot(vals, eff, marker="s")
        axes[1].set_ylabel("Wall-Plug Eff.")
        axes[2].plot(vals, spot, marker="^")
        axes[2].set_ylabel("Spot Size")
        for ax in axes:
            ax.set_xlabel(parameter)
            ax.grid(True)

        fig.tight_layout()
        path = Path("results") / self.project / "kpi_plots" / f"{label}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        self.last_kpi_plot = path
        return path

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

        for val in results:
            metrics[val]["wall_plug_efficiency"] = 0.0
            metrics[val]["spot_size"] = 0.0

        self.metrics[label] = metrics
        self.params[label] = parameter
        self._save_kpi_plot(label, parameter, metrics)
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
        *,
        solver: str = "random",
        n_generations: int = 25,
        pop_size: int = 40,
        hardware_constraint: Callable[[np.ndarray], bool] | None = None,
    ) -> List[Dict[str, float]]:
        """Run a Pareto search for yield versus spot size.

        Parameters
        ----------
        base_config:
            Starting configuration for each simulation.
        bounds:
            Mapping of parameter names to ``(min, max)`` bounds.
        n_samples:
            Number of random samples when ``solver='random'``.
        solver:
            Optimization backend.  ``"random"`` performs an unbiased search
            while ``"nsga2"`` executes a multi-objective genetic algorithm.
        n_generations:
            Generation count for ``"nsga2"`` runs.
        pop_size:
            Population size for ``"nsga2"`` runs.
        hardware_constraint:
            Optional predicate enforcing hardware limits for candidate vectors.
        """

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

        if solver == "random":
            pareto_params = random_pareto_search(evaluate, bounds, n_samples=n_samples)
            history: List[ConvergenceRecord] | None = None
        elif solver == "nsga2":
            pareto_params, history = nsga2(
                evaluate,
                bounds,
                n_generations=n_generations,
                pop_size=pop_size,
                constraint=hardware_constraint,
                return_history=True,
            )
        else:
            raise ValueError(f"Unknown solver '{solver}'")

        front: List[Dict[str, float]] = []
        for param_dict in pareto_params:
            params = tuple(param_dict[name] for name in names)
            y, s = eval_cache[params]
            front.append({**param_dict, "yield": y, "spot_size": s})

        if history:
            self._save_convergence_plot(history, solver)
        return front

    def _save_convergence_plot(
        self, history: List[ConvergenceRecord], solver: str
    ) -> Path:
        """Write convergence metrics to disk and track the plot path."""

        import matplotlib.pyplot as plt

        gens = [rec.generation for rec in history]
        best_y = [rec.best_yield for rec in history]
        min_s = [rec.min_spot_size for rec in history]

        fig, ax1 = plt.subplots()
        ax1.plot(gens, best_y, marker="o", color="tab:blue", label="Best Yield")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Yield", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(gens, min_s, marker="s", color="tab:red", label="Min Spot")
        ax2.set_ylabel("Spot Size", color="tab:red")
        fig.tight_layout()

        path = Path("results") / self.project / "convergence" / f"{solver}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        self.last_convergence_plot = path
        return path

    def overlay_metrics(
        self, path: str | Path | None = None, parameter: str | None = None
    ) -> Path:
        """Overlay KPI curves for stored sweeps.

        The generated figure contains yield, wall-plug efficiency and spot size
        panels.  When ``path`` is omitted the plot is written beneath
        ``results/<project>/kpi_plots/overlay.png``.
        """

        import matplotlib.pyplot as plt

        if parameter is None:
            params = {self.params.get(k, "") for k in self.metrics}
            parameter = params.pop() if len(params) == 1 else "parameter"

        if path is None:
            path = Path("results") / self.project / "kpi_plots" / "overlay.png"
        else:
            path = Path(path)
            if not path.is_absolute():
                path = Path("results") / self.project / "kpi_plots" / path
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, sharex=False, figsize=(12, 4))

        for label, metrics in self.metrics.items():
            vals = sorted(metrics.keys())
            y = [metrics[v].get("yield", 0.0) for v in vals]
            e = [
                metrics[v].get("wall_plug_efficiency", metrics[v].get("efficiency", 0.0))
                for v in vals
            ]
            s = [metrics[v].get("spot_size", 0.0) for v in vals]
            axes[0].plot(vals, y, label=label)
            axes[1].plot(vals, e, label=label)
            axes[2].plot(vals, s, label=label)

        axes[0].set_ylabel("Yield")
        axes[1].set_ylabel("Wall-Plug Eff.")
        axes[2].set_ylabel("Spot Size")
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
        computed ``yield``, ``pinch_time``, ``efficiency``,
        ``wall_plug_efficiency`` and ``spot_size`` metrics.

        Parameters
        ----------
        path:
            Destination path for the CSV file.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "label",
                    "parameter",
                    "yield",
                    "pinch_time",
                    "efficiency",
                    "wall_plug_efficiency",
                    "spot_size",
                ]
            )
            for label, metrics in self.metrics.items():
                for param_val, vals in metrics.items():
                    writer.writerow(
                        [
                            label,
                            param_val,
                            vals.get("yield", 0.0),
                            vals.get("pinch_time", 0.0),
                            vals.get("efficiency", 0.0),
                            vals.get("wall_plug_efficiency", 0.0),
                            vals.get("spot_size", 0.0),
                        ]
                    )
        return path

    def export_scene(self, path: str | Path) -> Path:
        """Export current metrics and parameters to ``path`` in JSON format."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "project": self.project,
            "metrics": self.metrics,
            "params": self.params,
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    def import_scene(self, path: str | Path) -> None:
        """Populate the manager from a JSON file produced by ``export_scene``."""

        data = json.loads(Path(path).read_text())
        self.project = data.get("project", self.project)
        metrics: Dict[str, Dict[float, Dict[str, float]]] = {}
        for label, vals in data.get("metrics", {}).items():
            metrics[label] = {float(k): v for k, v in vals.items()}
        self.metrics = metrics
        self.params = data.get("params", {})

    # ------------------------------------------------------------------
    # Geometry handling
    def import_geometry(self, label: str, path: str | Path) -> None:
        """Import a geometry file (STEP/STL/VTK) and store it under ``label``.

        The geometry is kept in-memory for later visualisation or transformation.
        Dependencies such as :mod:`trimesh` or :mod:`pyvista` are optional and
        only required when the corresponding file format is used.
        """

        p = Path(path)
        suffix = p.suffix.lower()
        geom: Any
        if suffix in {".stl", ".step", ".stp"}:
            try:
                import trimesh

                geom = trimesh.load(p)
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("trimesh is required for STEP/STL import") from exc
        elif suffix in {".vtk", ".vtu", ".vtp"}:
            try:
                import pyvista as pv

                geom = pv.read(p)
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("pyvista is required for VTK import") from exc
        else:
            raise ValueError(f"Unsupported geometry format: {suffix}")

        self.geometries[label] = geom

    def transform_geometry(
        self, label: str, translation: Tuple[float, float, float]
    ) -> None:
        """Apply a simple translation to a stored geometry."""

        geom = self.geometries.get(label)
        if geom is None:
            raise KeyError(label)
        try:
            geom.apply_translation(translation)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fall back for pyvista
            try:
                geom.translate(translation)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Geometry translation not supported") from exc

    def rotate_geometry(
        self, label: str, rotation: Tuple[float, float, float]
    ) -> None:
        """Rotate a stored geometry by ``(rx, ry, rz)`` degrees."""

        geom = self.geometries.get(label)
        if geom is None:
            raise KeyError(label)
        rx, ry, rz = rotation
        try:  # pragma: no cover - depends on optional trimesh
            import numpy as np
            import trimesh

            mat = trimesh.transformations.euler_matrix(
                np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz), "sxyz"
            )
            geom.apply_transform(mat)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
        try:  # pragma: no cover - depends on optional pyvista
            geom.rotate_x(rx, inplace=True)  # type: ignore[attr-defined]
            geom.rotate_y(ry, inplace=True)  # type: ignore[attr-defined]
            geom.rotate_z(rz, inplace=True)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Geometry rotation not supported") from exc

    def geometry_figure(self, label: str):  # pragma: no cover - simple wrapper
        """Return a Plotly figure visualising a stored geometry."""

        geom = self.geometries.get(label)
        if geom is None:
            raise KeyError(label)
        try:
            import plotly.graph_objects as go
            import numpy as np
        except Exception as exc:
            raise RuntimeError("plotly is required for geometry visualisation") from exc

        if hasattr(geom, "faces") and hasattr(geom, "vertices"):
            verts = np.asarray(geom.vertices)
            faces = np.asarray(geom.faces)
        elif hasattr(geom, "points") and hasattr(geom, "faces"):
            verts = np.asarray(geom.points)
            faces = np.asarray(geom.faces).reshape(-1, 4)[:, 1:]
        else:  # pragma: no cover - unknown object type
            raise RuntimeError("Unsupported geometry object")

        mesh = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2]
        )
        fig = go.Figure(data=[mesh])
        fig.update_layout(scene_aspectmode="data")
        return fig

    # ------------------------------------------------------------------
    # Circuit handling
    def add_component(self, name: str, node_a: str, node_b: str) -> None:
        """Add a circuit component between ``node_a`` and ``node_b``."""

        if self.circuit is None:
            raise RuntimeError("networkx is required for circuit wiring")
        self.circuit.add_edge(node_a, node_b, component=name)

    def circuit_figure(self):  # pragma: no cover - visual helper
        """Return a Plotly figure of the current circuit wiring."""

        if self.circuit is None:
            raise RuntimeError("networkx is required for circuit visualisation")
        try:
            import plotly.graph_objects as go
            import numpy as np
        except Exception as exc:
            raise RuntimeError("plotly is required for circuit visualisation") from exc

        pos = nx.spring_layout(self.circuit)
        x = []
        y = []
        for n in self.circuit.nodes:
            px, py = pos[n]
            x.append(px)
            y.append(py)
        edge_x = []
        edge_y = []
        for u, v in self.circuit.edges:
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])
        node_trace = go.Scatter(x=x, y=y, mode="markers+text", text=list(self.circuit.nodes))
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines")
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(showlegend=False)
        return fig


__all__ = ["ProjectManager"]
