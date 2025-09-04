from __future__ import annotations

"""Run a frozen benchmark case and compare against reference outputs.

This helper script executes a benchmark stored under ``benchmarks/<case>``
where each project provides ``inputs.json`` describing a simple RLC circuit
and ``expected.json`` containing reference waveforms along with tolerance
bands.  Results are written to ``Validation/<case>/`` with an overlay plot and
error metrics.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np

try:  # pragma: no cover - optional dependency
    import h5py  # type: ignore
except Exception:  # pragma: no cover - stubbed in tests
    h5py = None  # type: ignore[assignment]

try:  # pragma: no cover - matplotlib optional
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib may be absent
    plt = None  # type: ignore[assignment]

from dpf2.core.circuit import RLCCircuitSolver
from dpf2.core.bases import CouplingState


def run_benchmark(case: str, benchmark_dir: str = "benchmarks", output: str = "Validation") -> Dict[str, float]:
    """Execute ``case`` and overlay results against tolerance bands.

    Parameters
    ----------
    case:
        Name of the benchmark directory under ``benchmark_dir``.
    benchmark_dir:
        Root directory containing benchmark cases.  Defaults to ``benchmarks``.
    output:
        Destination directory for plots and metrics.  Results are written to
        ``output/case``.
    """

    project = Path(benchmark_dir) / case
    inputs_path = project / "inputs.json"
    expected_path = project / "expected.json"
    if not inputs_path.exists() or not expected_path.exists():
        raise FileNotFoundError(f"benchmark '{case}' not found under {benchmark_dir}")

    inputs = json.loads(inputs_path.read_text())
    expected = json.loads(expected_path.read_text())

    times = np.array(expected["time"])
    L = float(inputs["inductance"])
    C = float(inputs["capacitance"])
    circuit = RLCCircuitSolver(
        L_ext=L,
        R_ext=float(inputs.get("resistance", 0.0)),
        C_ext=C,
        V0=float(inputs["charging_voltage"]),
    )
    state = CouplingState(current=0.0, voltage=float(inputs["charging_voltage"]))
    max_dt = 0.5 * (L * C) ** 0.5
    dt = min(max_dt * 0.5, float(inputs["end_time"]) / 1000.0)
    num_steps = int(max(float(inputs["end_time"]) / dt, 1))
    for _ in range(num_steps):
        circuit.step(state, back_emf=0.0, dt=dt)
        state.current = circuit.currents[-1]
        state.voltage = circuit.voltages[-1]

    sim_time = np.array(circuit.time)
    sim_current = np.array(circuit.currents)
    sim_voltage = np.array(circuit.voltages)
    sim_yield = np.zeros_like(sim_time)

    sim_interp = {
        "current": np.interp(times, sim_time, sim_current),
        "voltage": np.interp(times, sim_time, sim_voltage),
        "neutron_yield": np.zeros_like(times),
    }

    metrics: Dict[str, float] = {}
    grades: Dict[str, str] = {}
    passed = True
    tol = expected.get("tolerance", {})

    def _grade(err: float, tol_val: float) -> str:
        if tol_val <= 0:
            return "N/A"
        ratio = err / tol_val
        if ratio <= 1:
            return "A"
        if ratio <= 2:
            return "B"
        if ratio <= 3:
            return "C"
        if ratio <= 4:
            return "D"
        return "F"

    for key, sim_vals in sim_interp.items():
        exp_vals = np.array(expected[key])
        err = float(np.max(np.abs(sim_vals - exp_vals)))
        metrics[key] = err
        g = _grade(err, float(tol.get(key, 0.0)))
        grades[key] = g
        passed = passed and g in {"A", "B"}

    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4, "N/A": 5}
    overall = max(grades.values(), key=lambda x: grade_order.get(x, 5)) if grades else "N/A"
    metrics["grades"] = grades
    metrics["overall_grade"] = overall
    metrics["passed"] = passed

    out_root = Path(output) / case
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if plt is not None:  # pragma: no cover - plotting optional
        fig, axes = plt.subplots(3, 1, figsize=(6, 8))
        fields = ["current", "voltage", "neutron_yield"]
        for ax, field in zip(axes, fields):
            exp_vals = np.array(expected[field])
            sim_vals = sim_interp[field]
            tol_val = float(tol.get(field, 0.0))
            ax.plot(times, exp_vals, label="expected")
            ax.plot(times, sim_vals, label="simulation")
            ax.fill_between(times, exp_vals - tol_val, exp_vals + tol_val, alpha=0.3, label="tolerance")
            ax.set_ylabel(field.replace("_", " "))
        axes[-1].set_xlabel("time (s)")
        axes[0].legend()
        axes[0].text(
            0.98,
            0.02,
            f"Grade: {overall}",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
        )
        fig.tight_layout()
        fig.savefig(out_root / "overlay.png")
        plt.close(fig)

    if h5py is not None:  # pragma: no cover - optional dependency
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
        cfg_hash = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
        with h5py.File(out_root / "results.h5", "w") as f:
            f.create_dataset("time", data=sim_time)
            f.create_dataset("current", data=sim_current)
            f.create_dataset("voltage", data=sim_voltage)
            f.create_dataset("neutron_yield", data=sim_yield)
            manifest = f.require_group("manifest")
            manifest.attrs["git_commit"] = commit
            manifest.attrs["config_hash"] = cfg_hash
            manifest.attrs["inputs"] = str(inputs_path)
            manifest.attrs["passed"] = passed

    return metrics


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Run a benchmark case")
    parser.add_argument("case", help="Benchmark case name under benchmarks/")
    parser.add_argument(
        "--benchmark-dir",
        default="benchmarks",
        help="Directory containing benchmark cases",
    )
    parser.add_argument(
        "--output",
        default="Validation",
        help="Directory where results will be written",
    )
    args = parser.parse_args()
    metrics = run_benchmark(args.case, args.benchmark_dir, args.output)
    status = "PASSED" if metrics.get("passed") else "FAILED"
    print(f"Benchmark {args.case} {status}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
