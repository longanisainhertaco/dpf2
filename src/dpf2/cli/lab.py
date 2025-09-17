"""Utilities for lab-mode reproducibility manifests."""
from __future__ import annotations
import json
import os
import platform
import subprocess
import sys
import random
from pathlib import Path
from typing import Sequence, Mapping

import click
import numpy as np

try:  # pragma: no cover - optional dependency
    import h5py
except Exception:  # pragma: no cover - h5py may be absent
    h5py = None  # type: ignore[assignment]


RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_H5_FILENAME = "run_manifest.h5"


def _code_hash() -> str:
    """Return current git commit hash if available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _environment() -> dict[str, object]:
    """Capture basic execution environment details."""
    hdf5_version = "unknown"
    if h5py is not None:
        try:  # pragma: no cover - h5py may be a stub
            hdf5_version = h5py.version.hdf5_version
        except Exception:
            hdf5_version = "unknown"
    mpi_version = (
        os.environ.get("MPI_VERSION")
        or os.environ.get("MPICH_VERSION")
        or os.environ.get("OMPI_VERSION")
        or "unknown"
    )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env": dict(os.environ),
        "container_hash": os.environ.get("CONTAINER_HASH", "unknown"),
        "compiler": os.environ.get("CC", "unknown"),
        "mpi": mpi_version,
        "hdf5": hdf5_version,
    }


def write_manifest(
    output_dir: str | Path,
    *,
    config_paths: Sequence[str] | None = None,
    config: Mapping[str, object] | None = None,
    ppc: int | None = None,
    seeds: Mapping[str, int] | None = None,
    warnings: Sequence[str] | None = None,
    datasets: Mapping[str, Mapping[str, object]] | None = None,
) -> Path:
    """Write a JSON manifest capturing reproducibility metadata.

    Parameters
    ----------
    output_dir:
        Directory where the manifest should be written.
    config_paths:
        Sequence of configuration files used for the run.
    config:
        Configuration dictionary for the executed run. When provided, this is
        embedded directly in the manifest to aid exact reproducibility.
    ppc:
        Particle-per-cell setting, if applicable.
    seeds:
        Mapping of RNG seeds (e.g. {"python": int, "numpy": int}). If not
        provided, the current RNG states are sampled.
    datasets:
        Optional mapping of dataset names to metadata dictionaries containing
        ``path``, ``doi`` and ``version``. Hashes of these datasets are stored
        in the manifest.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = {"python": random.getstate()[1][0]}
        try:
            seeds["numpy"] = int(np.random.get_state()[1][0])  # numpy<2.0
        except Exception:  # pragma: no cover - handle new RNG APIs
            try:
                rng = np.random.default_rng()
                # bit_generator state is large; capture a representative int
                seeds["numpy"] = int(rng.bit_generator.state["state"]["state"])
            except Exception:
                seeds["numpy"] = 0

    manifest = {
        "code_hash": _code_hash(),
        "random_seeds": dict(seeds),
        "particle_per_cell": ppc,
        "config_paths": [str(p) for p in (config_paths or [])],
        "environment": _environment(),
    }
    if config is not None:
        # ``config`` may contain non-serialisable objects; best effort to cast
        # via ``json.dumps`` when writing the manifest.
        try:
            manifest["config"] = json.loads(json.dumps(config))
        except Exception:
            manifest["config"] = {k: str(v) for k, v in config.items()}
    if warnings:
        manifest["warnings"] = list(warnings)
    dataset_meta = None
    if datasets:
        from ..io.manifest import capture_dataset_metadata
        dataset_meta = capture_dataset_metadata(datasets)
        manifest["datasets"] = dataset_meta

    path = out / RUN_MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2))

    if h5py is not None:  # pragma: no cover - only when h5py available
        h5_path = out / RUN_MANIFEST_H5_FILENAME
        with h5py.File(h5_path, "w") as h5:
            mgrp = h5.require_group("manifest")
            for key, value in manifest.items():
                if key == "datasets":
                    continue
                if isinstance(value, (dict, list)):
                    mgrp.attrs[key] = json.dumps(value)
                elif value is None:
                    mgrp.attrs[key] = "null"
                else:
                    mgrp.attrs[key] = value
            if dataset_meta:
                from ..io.manifest import write_hdf5_dataset_manifest
                write_hdf5_dataset_manifest(h5, dataset_meta)

    return path

__all__ = [
    "write_manifest",
    "RUN_MANIFEST_FILENAME",
    "RUN_MANIFEST_H5_FILENAME",
    "lab",
]


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------

from dataclasses import asdict
from ..core.config import DPFConfig
from ..core.simulation import DPFSimulation
from ..io.manifest import write_batch_manifest


@click.group()
def lab() -> None:
    """Lab-mode helper commands."""


@lab.command(name="run")
@click.option("--config", type=click.Path(exists=False), help="Path to config file")
@click.option("--shots", type=int, default=1, show_default=True, help="Number of shots")
@click.option("--output", type=click.Path(), default="output", help="Output directory")
def lab_run(config: str | None, shots: int, output: str) -> None:
    """Run a batch of jittered simulations."""

    cfg = DPFConfig.from_file(config) if config else DPFConfig()
    runs: list[dict[str, object]] = []
    for i in range(shots):
        py_seed = random.randint(0, 2**32 - 1)
        np_seed = random.randint(0, 2**32 - 1)
        random.seed(py_seed)
        rng = np.random.default_rng(np_seed)

        jittered_cfg, jit_vals = cfg.apply_jitter(rng)

        sim = DPFSimulation(jittered_cfg)
        shot_dir = Path(output) / f"shot_{i:03d}"
        sim.run(output_dir=shot_dir, seeds={"python": py_seed, "numpy": np_seed})
        write_manifest(
            shot_dir,
            config_paths=[config] if config else None,
            config=asdict(jittered_cfg),
            seeds={"python": py_seed, "numpy": np_seed},
        )

        run_info = {
            "shot": i,
            "seeds": {"python": py_seed, "numpy": np_seed},
            **jit_vals,
        }
        runs.append(run_info)

    write_batch_manifest(output, runs)
