from pathlib import Path
import hashlib
from typing import Mapping
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
    datasets: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> dict[str, dict[str, dict[str, str]]]:
    """Compute hashes and attach DOI/version for referenced datasets.

    Parameters
    ----------
    datasets:
        Mapping from dataset category ("atomic", "nuclear" or
        "material") to mappings of dataset name to information containing
        ``path``, ``doi`` and ``version`` entries.
    """

    result: dict[str, dict[str, dict[str, str]]] = {}
    for category, entries in datasets.items():
        group: dict[str, dict[str, str]] = {}
        for name, info in entries.items():
            path = info.get("path")
            doi = info.get("doi")
            version = info.get("version")
            if path is None or doi is None or version is None:
                raise ValueError("dataset metadata requires 'path', 'doi' and 'version'")
            p = Path(path)
            h = _hash_file(p)
            logger.info(
                "dataset %s/%s: hash=%s doi=%s version=%s",
                category,
                name,
                h,
                doi,
                version,
            )
            group[name] = {
                "path": str(p),
                "hash": h,
                "doi": str(doi),
                "version": str(version),
            }
        result[category] = group
    return result


def write_hdf5_dataset_manifest(
    h5file: "h5py.File", metadata: Mapping[str, Mapping[str, Mapping[str, str]]]
) -> None:  # pragma: no cover - thin wrapper
    """Embed dataset metadata in an HDF5 ``manifest`` group.

    Parameters
    ----------
    h5file:
        Open HDF5 file handle where the manifest should be written.
    metadata:
        Mapping from dataset category to metadata dictionaries produced by
        :func:`capture_dataset_metadata`.
    """

    manifest = h5file.require_group("manifest")
    dgrp = manifest.require_group("datasets")
    for category, datasets in metadata.items():
        cgrp = dgrp.require_group(category)
        for name, info in datasets.items():
            grp = cgrp.require_group(name)
            for key, value in info.items():
                grp.attrs[key] = value


__all__ = ["capture_dataset_metadata", "write_hdf5_dataset_manifest"]
