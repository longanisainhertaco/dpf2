"""Regression tests for parametric sweep utilities."""
from __future__ import annotations

from pathlib import Path

import pytest

from dpf2.core.config import DPFConfig
from dpf2.optimization.param_sweep import run_parametric_sweep, compute_sweep_metrics


pytest.importorskip("matplotlib")


def test_voltage_sweep_yield_monotonic(tmp_path: Path) -> None:
    cfg = DPFConfig()
    values = [10000.0, 20000.0, 30000.0]
    results = run_parametric_sweep(cfg, "charging_voltage", values, output_dir=tmp_path)
    metrics = compute_sweep_metrics(cfg, results, parameter="charging_voltage")
    yields = [metrics[v]["yield"] for v in values]
    assert yields == sorted(yields)


def test_pressure_sweep_optimal_s(tmp_path: Path) -> None:
    cfg = DPFConfig()
    values = [50.0, 100.0, 200.0]
    results = run_parametric_sweep(cfg, "initial_pressure", values, output_dir=tmp_path)
    metrics = compute_sweep_metrics(cfg, results, parameter="initial_pressure")
    best_yield_val = max(metrics, key=lambda v: metrics[v]["yield"])
    best_s_val = max(metrics, key=lambda v: metrics[v]["S"])
    assert best_yield_val == best_s_val
