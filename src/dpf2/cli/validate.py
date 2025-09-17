"""Validation command line interface for DPF2.

This module provides a small entry point to run a simulation and compare
its output against simple experimental traces using the
``ValidationSuite`` configuration. It is intentionally lightweight and
only supports the minimal data used in the tests and CI workflow.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:  # pragma: no cover - matplotlib may be absent
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback when matplotlib missing
    matplotlib = None
    plt = None  # type: ignore[assignment]
import numpy as np

from ..dpf_config import DPFConfig
from ..simulation_engine import SimulationEngine, SimulationResults
from ..validation_suite import (
    ValidationSuite,
    evaluate_benchmark,
    score_simulation,
)
from ..verification import VerificationPanel
from ..diagnostics.quality_dashboard import QualityDashboard
from ..scaling_laws import compare_to_scaling
from .lab import write_manifest


# ---------------------------------------------------------------------------
# Data handling helpers

def _build_validation_suite(dataset: str) -> ValidationSuite:
    """Create a :class:`ValidationSuite` for bundled validation data."""
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "data" / "validation" / dataset
    device_map = {"MJOLNIR": "LLNL-DPF", "LLNL_MJOLNIR": "LLNL-DPF"}
    return ValidationSuite(
        experiment_device_id=device_map.get(dataset, dataset),
        experiment_campaign_id="demo",
        dataset_directory=data_dir,
        dataset_format="csv",
        observable_file_map={
            "I(t)": Path("current.csv"),
            "V(t)": Path("voltage.csv"),
            "Yn": Path("neutron_yield.csv"),
        },
        observable_format_spec={
            "I(t)": {"time": "time", "value": "value"},
            "V(t)": {"time": "time", "value": "value"},
            "Yn": {"time": "time", "value": "value"},
        },
        validation_targets=["I(t)", "V(t)", "Yn"],
        observable_tolerances={"I(t)": 0.1, "V(t)": 0.1, "Yn": 0.2},
    )


def _load_experimental(vsuite: ValidationSuite) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load experimental observables from disk."""
    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, rel in vsuite.observable_file_map.items():
        path = vsuite.dataset_directory / rel
        arr = np.genfromtxt(path, delimiter=",", names=True)
        spec = vsuite.observable_format_spec.get(name, {}) if vsuite.observable_format_spec else {}
        t_col = spec.get("time", arr.dtype.names[0])
        v_col = spec.get("value", arr.dtype.names[1])
        data[name] = (arr[t_col], arr[v_col])
    return data


def _simulation_observables(res: SimulationResults) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract observables from the simulation results."""
    return {
        "I(t)": (res.time * 1e6, res.current / 1e3),
        "V(t)": (res.time * 1e6, res.voltage / 1e3),
        "Yn": (np.array([0.0]), np.array([res.neutron_yield])),
    }


def _plot_overlays(
    res: SimulationResults,
    exp: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Generate overlay plots of simulation vs. experiment."""
    if plt is None:  # pragma: no cover - matplotlib optional
        return
    sim = _simulation_observables(res)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (t_exp, v_exp) in exp.items():
        if name not in sim:
            continue
        t_sim, v_sim = sim[name]
        plt.figure()
        plt.plot(t_exp, v_exp, label="experiment")
        plt.plot(t_sim, v_sim, label="simulation")
        plt.xlabel("time [us]")
        plt.title(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{name.replace('/', '_')}_overlay.png")
        plt.close()


# ---------------------------------------------------------------------------
# Public API

def run_validation(
    config: Path,
    dataset: str,
    *,
    outdir: Path = Path("validation"),
    lab_mode: bool = False,
) -> bool:
    """Execute a simulation and validate against experimental data.

    Parameters
    ----------
    config:
        Path to a JSON/YAML :class:`DPFConfig` file.
    dataset:
        Identifier of the experimental dataset to use.
    outdir:
        Directory where overlay plots will be written.
    lab_mode:
        When ``True``, record a reproducibility manifest alongside outputs.

    Returns
    -------
    bool
        ``True`` if the validation passed according to the
        :class:`ValidationSuite` specification.
    """
    ds_path = Path(dataset)
    if ds_path.is_dir():
        exp = json.loads((ds_path / "expected.json").read_text())
        sim = json.loads((ds_path / "inputs.json").read_text())
        report = evaluate_benchmark(sim, exp, tolerances=exp.get("tolerance"))
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "benchmark_report.json").open("w") as fh:
            json.dump(report, fh, indent=2)
        for name, ok in report["checks"].items():
            status = "PASS" if ok else "FAIL"
            print(f"{name}: {status}")
        print("Benchmark passed" if report["passed"] else "Benchmark failed")
        return report["passed"]

    cfg = DPFConfig.from_file(config)
    engine = SimulationEngine(cfg)
    if lab_mode:
        seeds = {"python": random.getstate()[1][0]}
        try:
            seeds["numpy"] = int(np.random.get_state()[1][0])
        except Exception:
            try:
                rng = np.random.default_rng()
                seeds["numpy"] = int(rng.bit_generator.state["state"]["state"])
            except Exception:
                seeds["numpy"] = 0
    results = engine.run()

    vsuite = _build_validation_suite(dataset)
    exp = _load_experimental(vsuite)
    sim = _simulation_observables(results)
    tol_map = {
        "current": vsuite.observable_tolerances.get("I(t)", 1.0),
        "voltage": vsuite.observable_tolerances.get("V(t)", 1.0),
        "neutron_yield": vsuite.observable_tolerances.get("Yn", 1.0),
    }
    weight_map = None
    if vsuite.observable_weighting:
        weight_map = {
            "current": vsuite.observable_weighting.get("I(t)", 0.0),
            "voltage": vsuite.observable_weighting.get("V(t)", 0.0),
            "neutron_yield": vsuite.observable_weighting.get("Yn", 0.0),
        }
    report = score_simulation(
        sim,
        dataset,
        tol_map,
        resample_method=vsuite.resample_method or "interpolate",
        weights=weight_map,
        pass_threshold=vsuite.score_pass_threshold,
    )
    _plot_overlays(results, exp, outdir)

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "validation_report.json").open("w") as fh:
        json.dump(report, fh, indent=2)

    metrics = compare_to_scaling(results, vsuite.dataset_directory)
    if metrics:
        with (outdir / "scaling_report.json").open("w") as fh:
            json.dump(metrics, fh, indent=2)

    if lab_mode:
        ppc = getattr(getattr(cfg, "warpx_settings", None), "max_particles_per_cell", None)
        cfg_dict = (
            cfg.model_dump(mode="python") if hasattr(cfg, "model_dump") else cfg.__dict__
        )
        write_manifest(
            outdir,
            config_paths=[str(config)],
            config=cfg_dict,
            ppc=ppc,
            seeds=seeds,
        )

    for name, score in report["scores"].items():
        print(f"{name}: {score:.3f}")

    print(f"Overall score: {report['overall']:.3f}")
    print("Validation passed" if report["passed"] else "Validation failed")
    return report["passed"]


def run_verification(sizes: Iterable[int] = (16, 32, 64)) -> bool:
    """Execute verification problems and report observed orders."""
    dashboard = QualityDashboard()
    panel = VerificationPanel(quality=dashboard)
    results = {
        "brio_wu": panel.run_brio_wu(sizes),
        "orszag_tang": panel.run_orszag_tang(sizes),
        "mms_mhd": panel.run_mms_ideal_mhd(sizes),
    }
    for name, res in results.items():
        obs = res["observed_order"][-1] if res["observed_order"] else 0.0
        print(f"{name}: observed {obs:.2f} vs design 1.0")
    return all(r["passed"] for r in results.values())


def main(argv: Iterable[str] | None = None) -> None:
    args_list = list(argv) if argv is not None else None
    if args_list and len(args_list) > 0 and args_list[0] == "verify":
        parser = argparse.ArgumentParser(description="Run verification suite")
        parser.add_argument("--sizes", nargs="*", type=int, default=[16, 32, 64])
        opts = parser.parse_args(args_list[1:])
        run_verification(tuple(opts.sizes))
        return

    parser = argparse.ArgumentParser(
        description="Validate simulation against experimental data"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to DPF configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset identifier (e.g. PF1000)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("validation"),
        help="Directory to store overlay plots",
    )
    parser.add_argument(
        "--lab-mode",
        action="store_true",
        help="Record a reproducibility manifest alongside outputs",
    )
    args = parser.parse_args(args_list)
    run_validation(args.config, args.dataset, outdir=args.outdir, lab_mode=args.lab_mode)


if __name__ == "__main__":  # pragma: no cover
    main()
