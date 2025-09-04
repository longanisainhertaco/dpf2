"""Utilities for lab-mode reproducibility manifests."""
from __future__ import annotations

import json
import subprocess
import random
from pathlib import Path
from typing import Sequence, Mapping

import numpy as np


MANIFEST_FILENAME = "manifest.json"


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


def write_manifest(
    output_dir: str | Path,
    *,
    config_paths: Sequence[str] | None = None,
    ppc: int | None = None,
    seeds: Mapping[str, int] | None = None,
) -> Path:
    """Write a JSON manifest capturing reproducibility metadata.

    Parameters
    ----------
    output_dir:
        Directory where the manifest should be written.
    config_paths:
        Sequence of configuration files used for the run.
    ppc:
        Particle-per-cell setting, if applicable.
    seeds:
        Mapping of RNG seeds (e.g. {"python": int, "numpy": int}). If not
        provided, the current RNG states are sampled.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = {
            "python": random.getstate()[1][0],
            "numpy": int(np.random.get_state()[1][0]),
        }

    manifest = {
        "code_hash": _code_hash(),
        "random_seeds": dict(seeds),
        "particle_per_cell": ppc,
        "config_paths": [str(p) for p in (config_paths or [])],
    }

    path = out / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2))
    return path

__all__ = ["write_manifest", "MANIFEST_FILENAME"]
