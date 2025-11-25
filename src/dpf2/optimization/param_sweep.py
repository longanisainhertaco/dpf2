"""Utilities for evaluating full simulations over parameter sweeps.

This module drives :class:`~dpf2.core.simulation.DPFSimulation` for a set of
parameter values and records simple figures of merit.  For each sweep point the
simulation is executed, peak current is extracted and a crude neutron yield is
estimated from the peak current and device dimensions.  The dimensionless shock
parameter ``S = I/(a*p)`` is also computed to facilitate cross‑device
comparison.

Results are returned to the caller and additionally summarised in
``sweep_output/summary.json`` so that external tools can quickly inspect sweep
outputs without reloading large data files.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path

from typing import Dict, Iterable, List, Mapping
import json

try:  # pragma: no cover - optional dependency
    import h5py
except Exception:  # pragma: no cover - h5py may be stubbed
    h5py = None  # type: ignore[assignment]


from ..io.manifest import capture_dataset_metadata, write_hdf5_dataset_manifest
from ..simulation.openpmd_io import OpenPMDWriter


from ..core.config import DPFConfig
from ..core.simulation import DPFSimulation


# Each sweep result stores the raw traces and derived peaks
SweepResult = Dict[str, List[float] | float]


RUN_MANIFEST_FILENAME = "run_manifest.json"



def run_parametric_sweep(
    base_config: DPFConfig,
    parameter: str,
    values: Iterable[float],
    *,
    output_dir: str | Path = "sweep_output",
    lab_mode: bool = False,  # retained for API compatibility
    config_path: str | Path | None = None,
    emit_checkpoints: bool = False,
    emit_openpmd: bool = False,
    manifest: bool = False,
    datasets: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
) -> Dict[float, SweepResult]:
    """Execute full simulations for a set of parameter values.

    Parameters
    ----------
    base_config:
        Starting configuration for the sweep.  A shallow copy is made for each
        parameter value with the corresponding attribute replaced.
    parameter:
        Name of the configuration attribute to vary.
    values:
        Iterable of values for ``parameter``.
    output_dir, lab_mode, config_path:
        ``lab_mode`` and ``config_path`` are accepted for backward
        compatibility but unused.  ``output_dir`` controls where individual run
        outputs are written.  A ``summary.json`` file is produced containing
        the peak current, pinch time and yield for each sweep point.

    Returns
    -------
    Dict[float, SweepResult]

        Mapping of parameter values to result dictionaries containing ``time``,
        ``current`` and ``voltage`` traces along with derived metrics
        ``peak_current``, ``pinch_time`` and ``yield``.

    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[float, SweepResult] = {}
    summary: Dict[float, Dict[str, float]] = {}

    dataset_meta = capture_dataset_metadata(datasets) if datasets else None

    for val in values:

        cfg = replace(base_config, **{parameter: val})
        run_dir = out_dir / f"{parameter}_{val}"
        run_dir.mkdir(parents=True, exist_ok=True)
        sim = DPFSimulation(cfg)
        t0 = time.perf_counter()
        times, currents, voltages = sim.run(output_dir=None)
        runtime = time.perf_counter() - t0

        peak = max(currents) if currents else 0.0
        idx = currents.index(peak) if currents else 0
        pinch_time = times[idx] if times else 0.0

        # Simple proxy for neutron yield: I^2 scaled by geometry and pressure
        yield_val = 0.0
        if cfg.anode_radius > 0 and cfg.initial_pressure > 0:
            yield_val = (peak ** 2) / (cfg.anode_radius * cfg.initial_pressure)

        bank_energy = 0.5 * cfg.capacitance * (cfg.charging_voltage**2)
        wall_plug = yield_val / bank_energy if bank_energy > 0 else 0.0
        throughput = (yield_val / runtime) * 3600.0 if runtime > 0 else 0.0

        a = cfg.anode_radius
        p = cfg.initial_pressure
        if parameter == "anode_radius":
            a = float(val)
        if parameter == "initial_pressure":
            p = float(val)
        S = peak / (a * p) if a > 0 and p > 0 else 0.0

        results[float(val)] = {
            "time": [float(t) for t in times],
            "current": [float(i) for i in currents],
            "voltage": [float(v) for v in voltages],
            "peak_current": float(peak),
            "pinch_time": float(pinch_time),
            "yield": float(yield_val),
            "runtime_s": float(runtime),
            "yield_per_hour": float(throughput),
            "wall_plug_efficiency": float(wall_plug),
            "S": float(S),
        }
        summary[float(val)] = {
            "peak_current": float(peak),
            "pinch_time": float(pinch_time),
            "yield": float(yield_val),
            "runtime_s": float(runtime),
            "yield_per_hour": float(throughput),
            "wall_plug_efficiency": float(wall_plug),
            "S": float(S),

        }

        if emit_checkpoints:
            _write_checkpoint(
                run_dir,
                times,
                currents,
                voltages,
                datasets=dataset_meta,
            )
        if emit_openpmd:
            _write_openpmd(
                run_dir,
                times,
                currents,
                voltages,
                datasets=dataset_meta,
            )
        if manifest or lab_mode:
            from ..cli.lab import write_manifest

            write_manifest(
                run_dir,
                config_paths=[config_path] if config_path else None,
                config=asdict(cfg),
                ppc=None,
                seeds=None,
                datasets=datasets,
            )

    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    return results


def _write_checkpoint(
    run_dir: Path,
    times: List[float],
    currents: List[float],
    voltages: List[float],
    *,
    datasets: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
) -> None:
    """Persist a lightweight HDF5 checkpoint with traces and metadata."""

    if h5py is None:  # pragma: no cover - optional dependency path
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "checkpoint.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("time", data=times)
        h5.create_dataset("current", data=currents)
        h5.create_dataset("voltage", data=voltages)
        man = h5.require_group("manifest")
        man.attrs["format"] = "dpf2-timeseries"
        man.attrs["source"] = RUN_MANIFEST_FILENAME
        if datasets:
            write_hdf5_dataset_manifest(h5, datasets)


def _write_openpmd(
    run_dir: Path,
    times: List[float],
    currents: List[float],
    voltages: List[float],
    *,
    datasets: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
) -> None:
    """Emit an openPMD-compliant checkpoint capturing the time series."""

    try:
        writer = OpenPMDWriter(run_dir / "openpmd.h5", datasets=datasets)
    except Exception:  # pragma: no cover - optional dependency failure
        return

    try:
        writer.write_fields(0, {"current": currents, "voltage": voltages, "time": times})
    finally:  # pragma: no cover - best-effort cleanup
        writer.close()


def compute_sweep_metrics(
    base_config: DPFConfig,
    results: Dict[float, SweepResult],
    parameter: str | None = None,
) -> Dict[float, Dict[str, float]]:
    """Compute metrics such as yield and shock parameter ``S`` for sweep runs."""

    metrics: Dict[float, Dict[str, float]] = {}

    for val, data in results.items():
        peak = float(data.get("peak_current", 0.0))
        pinch = float(data.get("pinch_time", 0.0))
        yld = float(data.get("yield", 0.0))

        a = base_config.anode_radius
        p = base_config.initial_pressure
        if parameter == "anode_radius":
            a = float(val)
        if parameter == "initial_pressure":
            p = float(val)

        S = peak / (a * p) if a > 0 and p > 0 else 0.0
        metrics[val] = {
            "yield": yld,
            "pinch_time": pinch,
            "S": S,

            "efficiency": 0.0,
        }

    return metrics


def plot_metric_overlay(
    parameter: str,
    metrics: Dict[float, Dict[str, float]],
    path: str | Path,
) -> Path:
    """Plot yield, pinch time and efficiency against a swept parameter."""

    import matplotlib.pyplot as plt

    vals = sorted(metrics.keys())
    yields = [metrics[v]["yield"] for v in vals]
    pinch = [metrics[v].get("pinch_time", 0.0) for v in vals]
    effs = [metrics[v].get("efficiency", 0.0) for v in vals]
    y_lo = [metrics[v].get("yield_lo", metrics[v]["yield"]) for v in vals]
    y_hi = [metrics[v].get("yield_hi", metrics[v]["yield"]) for v in vals]
    p_lo = [metrics[v].get("pinch_time_lo", metrics[v].get("pinch_time", 0.0)) for v in vals]
    p_hi = [metrics[v].get("pinch_time_hi", metrics[v].get("pinch_time", 0.0)) for v in vals]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 9))

    axes[0].plot(vals, yields, marker="o")
    axes[0].fill_between(vals, y_lo, y_hi, color="C0", alpha=0.2)
    axes[0].set_ylabel("Yield")

    axes[1].plot(vals, pinch, marker="^")
    axes[1].fill_between(vals, p_lo, p_hi, color="C1", alpha=0.2)
    axes[1].set_ylabel("Pinch Time")

    axes[2].plot(vals, effs, marker="s")
    axes[2].set_ylabel("Efficiency")
    axes[2].set_xlabel(parameter)

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_yield_vs_S(metrics: Dict[float, Dict[str, float]], path: str | Path) -> Path:
    """Plot yield as a function of the shock parameter ``S``."""

    import matplotlib.pyplot as plt

    pairs = sorted(
        (
            m.get("S", 0.0),
            m.get("yield", 0.0),
            m.get("yield_lo", m.get("yield", 0.0)),
            m.get("yield_hi", m.get("yield", 0.0)),
        )
        for m in metrics.values()
    )
    s_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    y_lo = [p[2] for p in pairs]
    y_hi = [p[3] for p in pairs]
    plt.figure()
    plt.plot(s_vals, y_vals, marker="o")
    plt.fill_between(s_vals, y_lo, y_hi, color="C0", alpha=0.2)
    plt.xlabel("S")
    plt.ylabel("Yield")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    return path


def plot_yield_pressure_overlay(
    metric_sets: Dict[str, Dict[float, Dict[str, float]]],
    path: str | Path,
) -> Path:
    """Overlay yield vs. pressure curves for multiple sweeps."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for label, metrics in metric_sets.items():
        pressures = sorted(metrics.keys())
        yields = [metrics[p]["yield"] for p in pressures]
        plt.plot(pressures, yields, marker="o", label=label)
    plt.xlabel("Pressure")
    plt.ylabel("Yield")
    plt.legend()
    plt.savefig(path)
    plt.close()
    return path


__all__ = [
    "run_parametric_sweep",
    "compute_sweep_metrics",
    "plot_metric_overlay",
    "plot_yield_vs_S",
    "plot_yield_pressure_overlay",
    "SweepResult",
]
