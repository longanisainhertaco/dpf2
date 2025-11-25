"""Batch-mode helpers for sweeps and lightweight optimisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import click
import numpy as np

from ..core.config import DPFConfig
from ..optimization.param_sweep import (
    compute_sweep_metrics,
    run_parametric_sweep,
)
from ..io.manifest import write_batch_manifest


def _parse_values(values: Sequence[float], linspace: str | None) -> list[float]:
    if values:
        return [float(v) for v in values]
    if linspace:
        start, stop, count = linspace.split(":")
        return list(np.linspace(float(start), float(stop), int(count)))
    raise click.ClickException("Provide --value at least once or --linspace start:stop:count")


def _load_datasets(path: str | None) -> dict[str, dict[str, dict[str, object]]] | None:
    if path is None:
        return None
    meta_path = Path(path)
    try:
        payload = json.loads(meta_path.read_text())
    except Exception as exc:  # pragma: no cover - simple error path
        raise click.ClickException(f"Failed to read dataset manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException("Dataset manifest must be a JSON object")
    return payload  # type: ignore[return-value]


@click.group()
def batch() -> None:
    """Batch utilities for sweeps and optimisation."""


@batch.command(name="sweep")
@click.option("--config", type=click.Path(exists=False), help="Configuration file")
@click.option("--parameter", required=True, help="DPFConfig attribute to vary")
@click.option("--value", "values", type=float, multiple=True, help="Explicit parameter values")
@click.option(
    "--linspace",
    type=str,
    help="Generate evenly spaced values as start:stop:count",
)
@click.option("--output", type=click.Path(), default="sweep_output", help="Output directory")
@click.option("--emit-checkpoints", is_flag=True, help="Write HDF5 checkpoints for each shot")
@click.option("--openpmd", "emit_openpmd", is_flag=True, help="Emit openPMD checkpoints")
@click.option("--datasets", type=click.Path(), help="JSON mapping of dataset references")
@click.option("--manifest", is_flag=True, help="Write run manifests per sweep point")
@click.pass_context
def sweep(
    ctx: click.Context,
    config: str | None,
    parameter: str,
    values: Sequence[float],
    linspace: str | None,
    output: str,
    emit_checkpoints: bool,
    emit_openpmd: bool,
    datasets: str | None,
    manifest: bool,
) -> None:
    """Run a parameter sweep using :mod:`dpf2.optimization.param_sweep`."""

    cfg = DPFConfig.from_file(config) if config else DPFConfig()
    vals = _parse_values(values, linspace)
    datasets_meta = _load_datasets(datasets)
    manifest = manifest or ctx.obj.get("lab_mode", False)

    results = run_parametric_sweep(
        cfg,
        parameter,
        vals,
        output_dir=output,
        lab_mode=ctx.obj.get("lab_mode", False),
        config_path=config,
        emit_checkpoints=emit_checkpoints,
        emit_openpmd=emit_openpmd,
        manifest=manifest,
        datasets=datasets_meta,
    )
    metrics = compute_sweep_metrics(cfg, results, parameter=parameter)

    summary_rows = [
        {
            "value": v,
            "yield": m.get("yield", 0.0),
            "pinch_time": m.get("pinch_time", 0.0),
            "S": m.get("S", 0.0),
            "yield_per_hour": results[v].get("yield_per_hour", 0.0),
            "wall_plug_efficiency": results[v].get("wall_plug_efficiency", 0.0),
        }
        for v, m in metrics.items()
    ]
    write_batch_manifest(output, summary_rows)
    click.echo(json.dumps(summary_rows, indent=2))


@batch.command(name="optimize")
@click.option("--config", type=click.Path(exists=False), help="Configuration file")
@click.option("--parameter", required=True, help="Parameter to optimise")
@click.option("--lower", type=float, required=True, help="Lower bound for sweep")
@click.option("--upper", type=float, required=True, help="Upper bound for sweep")
@click.option("--steps", type=int, default=5, show_default=True, help="Number of evaluations")
@click.option("--output", type=click.Path(), default="opt_output", help="Output directory")
@click.pass_context
def optimize(
    ctx: click.Context,
    config: str | None,
    parameter: str,
    lower: float,
    upper: float,
    steps: int,
    output: str,
) -> None:
    """Coarse grid-search optimisation wrapper around sweeps."""

    cfg = DPFConfig.from_file(config) if config else DPFConfig()
    values = np.linspace(lower, upper, steps)
    results = run_parametric_sweep(
        cfg,
        parameter,
        values,
        output_dir=output,
        lab_mode=ctx.obj.get("lab_mode", False),
        config_path=config,
        emit_checkpoints=True,
        emit_openpmd=False,
        manifest=True,
    )
    metrics = compute_sweep_metrics(cfg, results, parameter=parameter)

    best_val, best_metrics = max(metrics.items(), key=lambda kv: kv[1].get("yield", 0.0))
    report = {
        "parameter": parameter,
        "best_value": float(best_val),
        "metrics": best_metrics,
        "throughput": results[best_val].get("yield_per_hour", 0.0),
        "wall_plug_efficiency": results[best_val].get("wall_plug_efficiency", 0.0),
    }
    write_batch_manifest(output, [report])
    click.echo(json.dumps(report, indent=2))


__all__ = ["batch"]
