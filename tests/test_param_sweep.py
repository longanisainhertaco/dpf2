"""Tests for parametric sweep utilities."""
from __future__ import annotations

from pathlib import Path

import h5py_stub  # noqa: F401  ensures stub is registered before importing modules

from dpf2.core.config import DPFConfig
from dpf2.optimization.param_sweep import (
    run_parametric_sweep,
    compute_sweep_metrics,
    plot_yield_vs_S,
)
import pytest
pytest.importorskip("matplotlib")


def test_parametric_sweep(tmp_path: Path) -> None:
    cfg = DPFConfig()
    values = [100.0, 150.0]
    results = run_parametric_sweep(cfg, "initial_pressure", values, output_dir=tmp_path)
    assert set(results.keys()) == set(values)
    for res in results.values():
        assert "yield" in res and "pinch_time" in res


def test_compute_sweep_metrics_includes_pinch(tmp_path: Path) -> None:
    cfg = DPFConfig()
    values = [100.0, 150.0]
    results = run_parametric_sweep(cfg, "initial_pressure", values, output_dir=tmp_path)
    metrics = compute_sweep_metrics(cfg, results, parameter="initial_pressure")
    for val in values:
        assert "pinch_time" in metrics[val]
        assert "yield_lo" in metrics[val]


def test_yield_vs_s_plot(tmp_path: Path) -> None:
    cfg = DPFConfig()
    values = [100.0, 150.0]
    results = run_parametric_sweep(cfg, "initial_pressure", values, output_dir=tmp_path)
    metrics = compute_sweep_metrics(cfg, results, parameter="initial_pressure")
    path = tmp_path / "ys.png"
    plot_yield_vs_S(metrics, path)
    assert path.exists()
    for val in values:
        assert "S" in metrics[val]
