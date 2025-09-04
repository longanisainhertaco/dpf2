from __future__ import annotations

"""Helpers for recording dataset provenance in run manifests."""

from pathlib import Path
import hashlib
from typing import Mapping
import logging


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
        h = _hash_file(Path(path))
        logger.info(
            "dataset %s: hash=%s doi=%s version=%s", name, h, doi, version
        )
        result[name] = {"hash": h, "doi": str(doi), "version": str(version)}
    return result


__all__ = ["capture_dataset_metadata"]
