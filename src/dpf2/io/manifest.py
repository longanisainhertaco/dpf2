from __future__ import annotations

"""Helpers for recording dataset provenance in run manifests."""

from pathlib import Path
import hashlib
import json
from typing import Mapping, Sequence, Mapping as MappingType
import logging

try:  # pragma: no cover - optional dependency
    import h5py  # type: ignore
except Exception:  # pragma: no cover - stubbed or missing
    h5py = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_dataset_metadata(
    datasets: Mapping[str, Mapping[str, object]]
) -> dict[str, dict[str, str]]:
    """Compute hashes and attach DOI/version for referenced datasets.

    Parameters
    ----------
    datasets:
        Mapping of dataset name to a mapping containing ``path``, ``doi`` and
        ``version`` entries.
    """

    result: dict[str, dict[str, str]] = {}
    for name, info in datasets.items():
        path = info.get("path")
        doi = info.get("doi")
        version = info.get("version")
        if path is None or doi is None or version is None:
            raise ValueError(
                "dataset metadata requires 'path', 'doi' and 'version'"
            )
        p = Path(path)
        h = _hash_file(p)
        logger.info(
            "dataset %s: hash=%s doi=%s version=%s", name, h, doi, version
        )
        result[name] = {
            "path": str(p),
            "hash": h,
            "doi": str(doi),
            "version": str(version),
        }
    return result


def write_hdf5_dataset_manifest(
    h5file: "h5py.File", metadata: Mapping[str, Mapping[str, str]]
) -> None:  # pragma: no cover - thin wrapper
    """Embed dataset metadata in an HDF5 ``manifest`` group.

    Parameters
    ----------
    h5file:
        Open HDF5 file handle where the manifest should be written.
    metadata:
        Mapping of dataset name to a mapping containing ``hash``, ``doi`` and
        ``version`` entries as produced by :func:`capture_dataset_metadata`.
    """

    manifest = h5file.require_group("manifest")
    dgrp = manifest.require_group("datasets")
    for name, info in metadata.items():
        grp = dgrp.require_group(name)
        for key, value in info.items():
            grp.attrs[key] = value


def write_batch_manifest(
    output_dir: str | Path, runs: Sequence[MappingType[str, object]]
) -> Path:
    """Write a manifest capturing run-to-run jitter variations.

    Parameters
    ----------
    output_dir:
        Directory where the manifest should be written.
    runs:
        Sequence describing each executed run. Each entry is serialised as-is
        and typically includes sampled parameter values and RNG seeds.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "batch_manifest.json"
    payload = {"runs": list(runs)}
    path.write_text(json.dumps(payload, indent=2))
    return path


__all__ = [
    "capture_dataset_metadata",
    "write_hdf5_dataset_manifest",
    "write_batch_manifest",
]
