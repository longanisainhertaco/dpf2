"""Utilities for managing lab-mode manifests via a simple API."""
from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

from ..cli.lab import _code_hash


def generate_manifest(config: Mapping[str, object], *, seed: int | None = None) -> dict[str, object]:
    """Generate a single manifest describing a run.

    Parameters
    ----------
    config:
        Configuration mapping used for the run.
    seed:
        Optional random seed. When omitted a random seed is sampled.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    return {
        "code_hash": _code_hash(),
        "inputs": dict(config),
        "random_seeds": {"python": int(seed)},
    }


def generate_manifest_bundle(
    configs: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    """Generate manifests for a batch of configurations."""
    bundle: list[dict[str, object]] = []
    for idx, cfg in enumerate(configs):
        seed = seeds[idx] if seeds and idx < len(seeds) else None
        bundle.append(generate_manifest(cfg, seed=seed))
    return bundle


def export_manifest_bundle(
    configs: Sequence[Mapping[str, object]],
    output: str | Path,
    *,
    seeds: Sequence[int] | None = None,
) -> Path:
    """Write a zip bundle containing inputs and manifests for each run.

    In addition to individual ``run_manifest_*.json`` and ``inputs_*.json``
    files, a small ``bundle_manifest.json`` index is written summarising the
    bundle contents.  This assists external HPC submission scripts in
    discovering the manifests without needing to scan the archive structure.

    Parameters
    ----------
    configs:
        Sequence of configuration dictionaries to include.
    output:
        Path to the zip file that should be written.
    seeds:
        Optional sequence of RNG seeds corresponding to ``configs``.
    """
    manifests = generate_manifest_bundle(configs, seeds=seeds)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, str]] = []
    with zipfile.ZipFile(out_path, "w") as z:
        for idx, (cfg, manifest) in enumerate(zip(configs, manifests)):
            cfg_name = f"inputs_{idx}.json"
            manifest_name = f"run_manifest_{idx}.json"
            z.writestr(cfg_name, json.dumps(cfg, indent=2))
            z.writestr(manifest_name, json.dumps(manifest, indent=2))
            index.append({"config": cfg_name, "manifest": manifest_name})
        z.writestr("bundle_manifest.json", json.dumps(index, indent=2))
    return out_path


__all__ = [
    "generate_manifest",
    "generate_manifest_bundle",
    "export_manifest_bundle",
]
