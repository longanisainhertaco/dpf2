"""Validation command line interface for DPF2.

This module provides a small entry point to run a simulation and compare
its output against simple experimental traces using the
``ValidationSuite`` configuration. It is intentionally lightweight and
only supports the minimal data used in the tests and CI workflow.
"""

from __future__ import annotations

import argparse
import json
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
from ..validation_suite import ValidationSuite
from ..scaling_laws import compare_to_scaling


# ---------------------------------------------------------------------------
# Data handling helpers

def _build_validation_suite(dataset: str) -> ValidationSuite:
    """Create a :class:`ValidationSuite` for bundled benchmark data."""
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "data" / "benchmarks" / dataset
    device_map = {"LLNL_MJOLNIR": "LLNL-DPF"}
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


def _score_observable(
    sim: Tuple[np.ndarray, np.ndarray],
    exp: Tuple[np.ndarray, np.ndarray],
    tolerance: float,
) -> float:
    """Compute a simple normalized RMSE score for a single observable."""
    exp_t, exp_v = exp
    sim_t, sim_v = sim
    sim_interp = np.interp(exp_t, sim_t, sim_v)
    rmse = float(np.sqrt(np.mean((sim_interp - exp_v) ** 2)))
    norm = np.max(np.abs(exp_v)) or 1.0
    return max(0.0, 1.0 - rmse / (norm * tolerance))


def _compute_scores(
    res: SimulationResults,
    vsuite: ValidationSuite,
    exp: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, float], float, bool]:
    """Calculate per-observable and aggregate scores."""
    sim = _simulation_observables(res)
    scores: Dict[str, float] = {}
    for obs in vsuite.validation_targets:
        if obs not in exp or obs not in sim:
            continue
        tol = vsuite.observable_tolerances.get(obs, 1.0)
        scores[obs] = _score_observable(sim[obs], exp[obs], tol)
    weights = vsuite.observable_weighting or {k: 1.0 for k in scores}
    total = sum(weights.values()) or 1.0
    overall = (
        sum(scores.get(k, 0.0) * weights.get(k, 0.0) for k in scores) / total
    )
    passed = overall >= vsuite.score_pass_threshold
    return scores, overall, passed


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

def run_validation(config: Path, dataset: str, *, outdir: Path = Path("validation")) -> bool:
    """Execute a simulation and validate against experimental data.

    Parameters
    ----------
    config:
        Path to a JSON/YAML :class:`DPFConfig` file.
    dataset:
        Identifier of the experimental dataset to use.
    outdir:
        Directory where overlay plots will be written.

    Returns
    -------
    bool
        ``True`` if the validation passed according to the
        :class:`ValidationSuite` specification.
    """
    cfg = DPFConfig.from_file(config)
    engine = SimulationEngine(cfg)
    results = engine.run()

    vsuite = _build_validation_suite(dataset)
    exp = _load_experimental(vsuite)
    scores, overall, passed = _compute_scores(results, vsuite, exp)
    _plot_overlays(results, exp, outdir)

    metrics = compare_to_scaling(results, vsuite.dataset_directory)
    if metrics:
        outdir.mkdir(parents=True, exist_ok=True)
        with (outdir / "scaling_report.json").open("w") as fh:
            json.dump(metrics, fh, indent=2)

    for name, score in scores.items():
        print(f"{name}: {score:.3f}")
        
    print(f"Overall score: {overall:.3f}")
    print("Validation passed" if passed else "Validation failed")
    return passed


def main(argv: Iterable[str] | None = None) -> None:
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
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_validation(args.config, args.dataset, outdir=args.outdir)


if __name__ == "__main__":  # pragma: no cover
    main()
