from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Tuple

import click
import numpy as np

try:  # pragma: no cover - matplotlib optional
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib may be absent
    plt = None  # type: ignore[assignment]

from ..benchmark_matching import BenchmarkMatching


def _load_config(path: Path) -> BenchmarkMatching:
    """Load a BenchmarkMatching configuration from ``path``."""
    text = path.read_text()
    try:  # pragma: no cover - prefer pydantic API when available
        return BenchmarkMatching.model_validate_json(text)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback for stubbed pydantic
        data = json.loads(text)
        return BenchmarkMatching(**data)


def _load_waveform(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return time and value arrays from a CSV file."""
    with path.open() as fh:
        reader = csv.DictReader(fh)
        names = reader.fieldnames or ["time", "value"]
        t_vals, v_vals = [], []
        for row in reader:
            t_vals.append(float(row[names[0]]))
            v_vals.append(float(row[names[1]]))
    return np.array(t_vals), np.array(v_vals)


@click.command("match-benchmark")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to BenchmarkMatching configuration file",
)
@click.option(
    "--simulation",
    "sim_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Simulation waveform CSV to compare against the benchmark",
)
@click.option(
    "--output",
    "outdir",
    type=click.Path(file_okay=False),
    default="benchmark_reports",
    show_default=True,
    help="Directory where reports will be written",
)
def match_benchmark(config_path: str, sim_path: str, outdir: str) -> None:
    """Compare simulation outputs against benchmark traces."""
    cfg = _load_config(Path(config_path))

    bench_path = cfg.benchmark_trace_path
    if bench_path is None:
        raise click.ClickException("benchmark_trace_path must be provided in config")
    bench_path = Path(bench_path)
    if not bench_path.is_absolute():
        bench_path = Path(config_path).parent / bench_path

    bench_t, bench_v = _load_waveform(bench_path)
    sim_t, sim_v = _load_waveform(Path(sim_path))

    sim_interp = np.interp(bench_t, sim_t, sim_v)
    err = sim_interp - bench_v
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_ref = float(np.max(np.abs(bench_v))) or 1.0
    rmse_pct = rmse / max_ref * 100.0
    passed = (
        rmse_pct <= cfg.waveform_tolerance
        if cfg.waveform_tolerance is not None
        else False
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "dataset": cfg.dataset_id,
        "rmse": rmse,
        "rmse_percent": rmse_pct,
        "tolerance_percent": cfg.waveform_tolerance,
        "passed": passed,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if plt is not None:  # pragma: no cover - plotting optional
        fig, ax = plt.subplots()
        ax.plot(bench_t, bench_v, label="benchmark")
        ax.plot(bench_t, sim_interp, label="simulation")
        ax.set_xlabel(f"time ({cfg.benchmark_time_unit or 's'})")
        label = cfg.benchmark_fields[0] if cfg.benchmark_fields else "value"
        ax.set_ylabel(label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "comparison.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        text = (
            f"Dataset: {cfg.dataset_id}\n"
            f"RMSE: {rmse:.3g}\n"
            f"RMSE (%): {rmse_pct:.2f}\n"
            f"Passed: {passed}"
        )
        ax.text(0.0, 1.0, text, va="top")
        fig.savefig(out / "report.pdf")
        plt.close(fig)

    html = (
        "<html><body>"
        f"<h1>Benchmark {cfg.dataset_id}</h1>"
        f"<p>RMSE: {rmse:.3g}</p>"
        f"<p>RMSE (%): {rmse_pct:.2f}</p>"
        f"<p>Passed: {passed}</p>"
        "</body></html>"
    )
    (out / "report.html").write_text(html)


__all__ = ["match_benchmark"]
