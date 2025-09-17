#!/usr/bin/env python3
"""Run pinch validation against benchmark datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dpf2.dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine
from dpf2.validation_suite import (
    load_pinch_dataset,
    compute_pinch_error_metrics,
)

TOLERANCES = {
    "current_rmse": 50.0,
    "voltage_rmse": 5.0,
    "neutron_yield_rmse": 1e9,
    "radius_rmse": 1.0,
    "current_t_peak": 2.0,
    "voltage_t_peak": 2.0,
    "radius_t_peak": 2.0,
    "energy_diff": 1e5,
}


def run_dataset(dataset_dir: Path, out_dir: Path) -> dict[str, float]:
    cfg = DPFConfig.with_defaults()
    engine = SimulationEngine(cfg)
    results = engine.run()

    t = results.time
    current = results.current
    radius = results.radius

    circuit = engine._setup_circuit()
    dIdt = np.gradient(current, t)
    voltage = (
        circuit.circuit.V0 - circuit.circuit.R * current - circuit.circuit.L * dIdt
    )
    neutron = np.column_stack([t, np.full_like(t, results.neutron_yield)])

    sim_outputs = {
        "current": (t, current),
        "voltage": (t, voltage),
        "neutron_yield": (neutron[:, 0], neutron[:, 1]),
        "radius": (t, radius),
    }

    metrics = compute_pinch_error_metrics(sim_outputs, dataset_dir, TOLERANCES)

    ref = load_pinch_dataset(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (rt, rv) in ref.items():
        st, sv = sim_outputs[name]
        plt.figure()
        plt.plot(rt, rv, label="reference")
        plt.plot(st, sv, label="simulation")
        plt.xlabel("time [au]")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset_dir.name}_{name}.png")
        plt.close()

    if not metrics["passed"]:
        raise SystemExit("Validation failed: " + json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=Path,
        default=Path("data/benchmarks"),
        help="Directory containing benchmark subdirectories.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Validation"),
        help="Output directory for comparison plots.",
    )
    args = parser.parse_args()

    metrics: dict[str, dict[str, float]] = {}
    for dataset in sorted(p for p in args.datasets.iterdir() if p.is_dir()):
        metrics[dataset.name] = run_dataset(dataset, args.out)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
