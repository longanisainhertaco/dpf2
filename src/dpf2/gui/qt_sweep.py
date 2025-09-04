from __future__ import annotations

"""PyQt-based GUI for basic parametric sweeps.

The interface exposes controls for charging voltage and initial gas pressure
and allows quick parametric sweeps over either quantity.  After each sweep a
plot of yield versus the swept parameter is displayed using :mod:`matplotlib`.

A simple multi-objective (yield vs. spot size) optimization is also provided via
:func:`dpf2.gui.project_manager.ProjectManager.pareto_search`.

This module only depends on :mod:`PyQt5` at runtime; if that optional
dependency is missing an informative :class:`RuntimeError` is raised when
attempting to launch the GUI.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from .project_manager import ProjectManager
from ..core.config import DPFConfig

try:  # pragma: no cover - optional GUI dependency
    from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QDoubleSpinBox,
        QPushButton,
    )
except Exception:  # pragma: no cover - allow import when PyQt missing
    QApplication = None  # type: ignore


def _ensure_qt() -> None:
    """Raise if :mod:`PyQt5` is unavailable."""

    if QApplication is None:  # pragma: no cover - executed only when missing
        raise RuntimeError("PyQt5 is required for the Qt GUI")


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

        self.pm = ProjectManager()

        # Connect callbacks
        self.btn_sweep_v.clicked.connect(
            lambda: self._run_sweep("charging_voltage", [0.8, 1.0, 1.2])
        )
        self.btn_sweep_p.clicked.connect(
            lambda: self._run_sweep("initial_pressure", [0.5, 1.0, 1.5])
        )
        self.btn_overlay.clicked.connect(self._overlay_runs)
        self.btn_export.clicked.connect(self._export_metrics)
        self.btn_pareto.clicked.connect(self._pareto_search)

    # ------------------------------------------------------------------
    def _config(self) -> DPFConfig:
        cfg = DPFConfig()
        cfg.charging_voltage = float(self.voltage.value())
        cfg.initial_pressure = float(self.pressure.value())
        return cfg

    def _run_sweep(self, param: str, factors: List[float]) -> None:
        cfg = self._config()
        base = getattr(cfg, param)
        values = [base * f for f in factors]
        label = f"{param}_{len(self.pm.metrics)}"
        metrics = self.pm.run_sweep(label, cfg, param, values)
        vals = sorted(metrics.keys())
        plt.figure()
        plt.plot(vals, [metrics[v]["yield"] for v in vals], marker="o")
        plt.xlabel(param)
        plt.ylabel("Yield")
        plt.tight_layout()
        plt.show()

    def _overlay_runs(self) -> None:
        path = self.pm.overlay_metrics(Path("overlay.png"))
        img = plt.imread(path)
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _export_metrics(self) -> None:
        self.pm.export_metrics(Path("metrics.csv"))

    def _pareto_search(self) -> None:
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
        front = self.pm.pareto_search(cfg, bounds, n_samples=20)
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


__all__ = ["launch"]
