from __future__ import annotations

"""PyQt-based GUI for basic parametric sweeps.

The interface exposes controls for charging voltage and initial gas pressure
and allows quick parametric sweeps over either quantity.  After each sweep a
plot of yield versus the swept parameter is displayed using :mod:`matplotlib`.

In addition to the GUI, a small CLI is provided for headless parametric sweeps
that reuses the same helpers.  Both interfaces allow specifying explicit
parameter ranges rather than the fixed factors previously used.

A simple multi-objective (yield vs. spot size) optimization is also provided via
:func:`dpf2.gui.project_manager.ProjectManager.pareto_search`.

This module only depends on :mod:`PyQt5` at runtime for the GUI portion; if that
optional dependency is missing an informative :class:`RuntimeError` is raised
when attempting to launch the GUI.
"""

from pathlib import Path
from typing import Dict, Iterable, List

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .project_manager import ProjectManager
from ..core.config import DPFConfig
from ..optimization.param_sweep import (
    compute_sweep_metrics,
    plot_metric_overlay,
    run_parametric_sweep,
)

try:  # pragma: no cover - optional GUI dependency
    from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QDoubleSpinBox,
        QSpinBox,
        QPushButton,
        QLineEdit,
    )
except Exception:  # pragma: no cover - allow import when PyQt missing
    QApplication = None  # type: ignore


def _ensure_qt() -> None:
    """Raise if :mod:`PyQt5` is unavailable."""

    if QApplication is None:  # pragma: no cover - executed only when missing
        raise RuntimeError("PyQt5 is required for the Qt GUI")


def plot_yield_vs_S_gv(
    metrics: Dict[float, Dict[str, float]], path: Path, gv_S: float = 2.0
) -> Path:
    """Plot yield versus shock parameter ``S`` and mark GV prediction.

    Points whose ``S`` deviates from the ``gv_S`` prediction are coloured red.
    The resulting image is written to ``path``.
    """

    s_vals = [m.get("S", 0.0) for m in metrics.values()]
    y_vals = [m.get("yield", 0.0) for m in metrics.values()]
    colors = ["red" if abs(s - gv_S) > 0.5 else "blue" for s in s_vals]
    plt.figure()
    plt.scatter(s_vals, y_vals, c=colors)
    plt.axvline(gv_S, color="k", linestyle="--", label="GV prediction")
    plt.xlabel("S")
    plt.ylabel("Yield")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    return path


def plot_yield_vs_param(
    parameter: str,
    metrics: Dict[float, Dict[str, float]],
    path: Path,
    gv_S: float = 2.0,
) -> Path:
    """Plot yield against a swept parameter and highlight optimal ``S``.

    The point with the best yield is emphasised and annotated with its
    corresponding ``S`` value.  Additionally, points whose ``S`` deviates
    markedly from the ``gv_S`` target are coloured red to make sub-optimal
    operating conditions obvious at a glance.
    """

    vals = sorted(metrics.keys())
    y_vals = [metrics[v].get("yield", 0.0) for v in vals]
    s_vals = [metrics[v].get("S", 0.0) for v in vals]
    colors = ["red" if abs(s - gv_S) > 0.5 else "blue" for s in s_vals]

    # Determine the best yield and corresponding parameter/S values
    best_idx = max(range(len(vals)), key=lambda i: y_vals[i]) if vals else 0
    best_val = vals[best_idx] if vals else 0.0
    best_yield = y_vals[best_idx] if vals else 0.0
    best_S = s_vals[best_idx] if vals else 0.0

    plt.figure()
    plt.scatter(vals, y_vals, c=colors)
    plt.scatter(
        [best_val],
        [best_yield],
        c="gold",
        edgecolors="black",
        zorder=3,
        label=f"best S={best_S:.2f}",
    )
    plt.xlabel(parameter)
    plt.ylabel("Yield")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    return path


class SweepPanel(QWidget):
    """Simple widget displaying overlays of sweep metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def overlay(self, managers: Dict[str, ProjectManager], metric: str = "yield") -> None:
        """Overlay ``metric`` versus parameter for all managed sweeps."""

        self.ax.clear()
        x_label = "parameter"
        for proj, pm in managers.items():
            for label, metrics in pm.metrics.items():
                vals = sorted(metrics.keys())
                y = [metrics[v].get(metric, 0.0) for v in vals]
                self.ax.plot(vals, y, label=f"{proj}:{label}")
                if pm.params.get(label):
                    x_label = pm.params[label]
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(metric.replace("_", " ").title())
        self.ax.legend()
        self.canvas.draw_idle()


class _SweepWindow(QWidget):
    """Main window for the sweep GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DPF2 Sweeps")
        layout = QVBoxLayout(self)

        self.voltage = QDoubleSpinBox()
        self.voltage.setRange(5_000, 30_000)
        self.voltage.setValue(10_000)
        self.voltage.setSuffix(" V")

        self.pressure = QDoubleSpinBox()
        self.pressure.setRange(0.1, 5.0)
        self.pressure.setSingleStep(0.1)
        self.pressure.setValue(1.0)
        self.pressure.setSuffix(" Torr")

        v_row = QHBoxLayout()
        v_row.addWidget(QLabel("Charging Voltage"))
        v_row.addWidget(self.voltage)
        layout.addLayout(v_row)

        p_row = QHBoxLayout()
        p_row.addWidget(QLabel("Initial Pressure"))
        p_row.addWidget(self.pressure)
        layout.addLayout(p_row)

        # Project name input for multi-project overlays
        self.project_edit = QLineEdit("qt")
        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("Project"))
        proj_row.addWidget(self.project_edit)
        layout.addLayout(proj_row)

        # Voltage sweep range controls
        self.v_min = QDoubleSpinBox()
        self.v_min.setRange(5_000, 30_000)
        self.v_min.setValue(8_000)
        self.v_min.setSuffix(" V")
        self.v_max = QDoubleSpinBox()
        self.v_max.setRange(5_000, 30_000)
        self.v_max.setValue(12_000)
        self.v_max.setSuffix(" V")
        self.v_steps = QSpinBox()
        self.v_steps.setRange(2, 20)
        self.v_steps.setValue(3)
        v_range = QHBoxLayout()
        v_range.addWidget(QLabel("V min"))
        v_range.addWidget(self.v_min)
        v_range.addWidget(QLabel("V max"))
        v_range.addWidget(self.v_max)
        v_range.addWidget(QLabel("steps"))
        v_range.addWidget(self.v_steps)
        layout.addLayout(v_range)

        # Pressure sweep range controls
        self.p_min = QDoubleSpinBox()
        self.p_min.setRange(0.1, 5.0)
        self.p_min.setSingleStep(0.1)
        self.p_min.setValue(0.5)
        self.p_min.setSuffix(" Torr")
        self.p_max = QDoubleSpinBox()
        self.p_max.setRange(0.1, 5.0)
        self.p_max.setSingleStep(0.1)
        self.p_max.setValue(1.5)
        self.p_max.setSuffix(" Torr")
        self.p_steps = QSpinBox()
        self.p_steps.setRange(2, 20)
        self.p_steps.setValue(3)
        p_range = QHBoxLayout()
        p_range.addWidget(QLabel("P min"))
        p_range.addWidget(self.p_min)
        p_range.addWidget(QLabel("P max"))
        p_range.addWidget(self.p_max)
        p_range.addWidget(QLabel("steps"))
        p_range.addWidget(self.p_steps)
        layout.addLayout(p_range)

        btn_row = QHBoxLayout()
        self.btn_sweep_v = QPushButton("Sweep Voltage")
        self.btn_sweep_p = QPushButton("Sweep Pressure")
        self.btn_pareto = QPushButton("Pareto Search")
        self.btn_overlay = QPushButton("Overlay Runs")
        self.btn_export = QPushButton("Export Metrics")
        for b in [
            self.btn_sweep_v,
            self.btn_sweep_p,
            self.btn_overlay,
            self.btn_pareto,
            self.btn_export,
        ]:
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        # Managers for each project
        self.managers: Dict[str, ProjectManager] = {}

        # Embedded panel for overlay plots
        self.panel = SweepPanel()
        layout.addWidget(self.panel)

        # Connect callbacks
        self.btn_sweep_v.clicked.connect(self._sweep_voltage)
        self.btn_sweep_p.clicked.connect(self._sweep_pressure)
        self.btn_overlay.clicked.connect(self._overlay_runs)
        self.btn_export.clicked.connect(self._export_metrics)
        self.btn_pareto.clicked.connect(self._pareto_search)

    # ------------------------------------------------------------------
    def _config(self) -> DPFConfig:
        cfg = DPFConfig()
        cfg.charging_voltage = float(self.voltage.value())
        cfg.initial_pressure = float(self.pressure.value())
        return cfg

    def _pm(self) -> ProjectManager:
        proj = self.project_edit.text() or "qt"
        return self.managers.setdefault(proj, ProjectManager(project=proj))

    def _sweep_voltage(self) -> None:
        vals = np.linspace(
            float(self.v_min.value()),
            float(self.v_max.value()),
            int(self.v_steps.value()),
        )
        self._run_sweep("charging_voltage", vals)

    def _sweep_pressure(self) -> None:
        vals = np.linspace(
            float(self.p_min.value()),
            float(self.p_max.value()),
            int(self.p_steps.value()),
        )
        self._run_sweep("initial_pressure", vals)

    def _run_sweep(self, param: str, values: Iterable[float]) -> None:
        pm = self._pm()
        cfg = self._config()
        label = f"{param}_{len(pm.metrics)}"
        metrics = pm.run_sweep(label, cfg, param, list(values))
        if pm.last_kpi_plot:
            img = plt.imread(pm.last_kpi_plot)
            plt.figure()
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        y_path = plot_yield_vs_param(
            param,
            metrics,
            Path("results") / pm.project
            / f"yield_vs_{param}"
            / f"{label}.png",
        )
        img = plt.imread(y_path)
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        if param == "initial_pressure":
            s_path = plot_yield_vs_S_gv(
                metrics,
                Path("results") / pm.project / "yield_vs_S" / f"{label}.png",
            )
            img = plt.imread(s_path)
            plt.figure()
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            best_val, best_metrics = max(
                metrics.items(), key=lambda kv: kv[1].get("yield", 0.0)
            )
            opt_S = best_metrics.get("S", 0.0)
            gv = 2.0
            print(f"Optimal S = {opt_S:.2f} at P={best_val}")
            if abs(opt_S - gv) > 1e-6:
                print(f"Deviation from GV ({gv}) = {opt_S - gv:+.2f}")
        self._overlay_runs()

    def _overlay_runs(self) -> None:
        self.panel.overlay(self.managers)

    def _export_metrics(self) -> None:
        pm = self._pm()
        pm.export_metrics(Path("metrics.csv"))

    def _pareto_search(self) -> None:
        pm = self._pm()
        cfg = self._config()
        bounds = {
            "charging_voltage": (
                cfg.charging_voltage * 0.8,
                cfg.charging_voltage * 1.2,
            ),
            "initial_pressure": (
                cfg.initial_pressure * 0.5,
                cfg.initial_pressure * 1.5,
            ),
        }
        front = pm.pareto_search(cfg, bounds, n_samples=20)
        plt.figure()
        plt.scatter(
            [p["spot_size"] for p in front],
            [p["yield"] for p in front],
        )
        plt.xlabel("Spot Size")
        plt.ylabel("Yield")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------

def launch() -> None:
    """Launch the Qt-based sweep GUI."""

    _ensure_qt()
    app = QApplication([])
    win = _SweepWindow()
    win.show()
    app.exec_()


def main(argv: List[str] | None = None) -> None:
    """Entry point for simple CLI-driven sweeps."""

    parser = argparse.ArgumentParser(description="Run a parametric sweep")
    parser.add_argument("--gui", action="store_true", help="launch the Qt GUI")
    parser.add_argument(
        "--param",
        choices=["charging_voltage", "initial_pressure"],
        help="Configuration parameter to sweep",
    )
    parser.add_argument("--min", type=float, help="Minimum value for the sweep")
    parser.add_argument("--max", type=float, help="Maximum value for the sweep")
    parser.add_argument("--steps", type=int, default=3, help="Number of sweep points")
    parser.add_argument(
        "--output",
        type=str,
        default="cli_sweep",
        help="Directory for sweep outputs",
    )
    args = parser.parse_args(argv)

    if args.gui or not args.param or args.min is None or args.max is None:
        launch()
        return

    cfg = DPFConfig()
    values = np.linspace(args.min, args.max, args.steps)
    results = run_parametric_sweep(cfg, args.param, values, output_dir=args.output)
    metrics = compute_sweep_metrics(cfg, results, args.param)
    plot_metric_overlay(
        args.param, metrics, Path(args.output) / f"{args.param}_metrics.png"
    )
    plot_yield_vs_param(
        args.param, metrics, Path(args.output) / f"yield_vs_{args.param}.png"
    )
    if args.param == "initial_pressure":
        plot_yield_vs_S_gv(metrics, Path(args.output) / "yield_vs_S.png")
        best_val, best_metrics = max(
            metrics.items(), key=lambda kv: kv[1].get("yield", 0.0)
        )
        opt_S = best_metrics.get("S", 0.0)
        gv = 2.0
        print(f"Optimal S = {opt_S:.2f} at P={best_val}")
        if abs(opt_S - gv) > 1e-6:
            print(f"Deviation from GV ({gv}) = {opt_S - gv:+.2f}")
    else:
        best_val = max(metrics, key=lambda v: metrics[v].get("yield", 0.0))
        print(f"Optimal {args.param} = {best_val}")


__all__ = ["launch", "main"]


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
