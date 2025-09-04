"""Tests for parametric sweep utilities."""
from __future__ import annotations

from pathlib import Path

import h5py_stub  # noqa: F401  ensures stub is registered before importing modules

from dpf2.core.config import DPFConfig
from dpf2.optimization.param_sweep import (
    plot_sweep_results,
    run_parametric_sweep,
    compute_sweep_metrics,
)
import pytest
pytest.importorskip("matplotlib")


def test_parametric_sweep(tmp_path: Path) -> None:
    cfg = DPFConfig()
    values = [10_000.0, 15_000.0]
    results = run_parametric_sweep(cfg, "charging_voltage", values, output_dir=tmp_path)
    assert set(results.keys()) == set(values)
    plot_path = tmp_path / "plot.png"
    plot_sweep_results("charging_voltage", results, plot_path)
    assert plot_path.exists()


def test_compute_sweep_metrics_includes_pinch(tmp_path: Path) -> None:
    cfg = DPFConfig()
    values = [10_000.0, 15_000.0]
    results = run_parametric_sweep(cfg, "charging_voltage", values, output_dir=tmp_path)
    metrics = compute_sweep_metrics(cfg, results)
    for val in values:
        assert "pinch_time" in metrics[val]
