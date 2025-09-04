from __future__ import annotations

"""Metadata loaders for chemistry datasets.

These helpers validate that required provenance information is present in
sidecar JSON metadata files distributed with ADAS or LXCat tables.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
import logging


@dataclass
class DatasetMetadata:
    """Minimal provenance for a tabulated data set."""

    doi: str
    version: str


logger = logging.getLogger(__name__)


def _require_metadata_fields(data: Mapping[str, object]) -> DatasetMetadata:
    """Validate required provenance fields and build :class:`DatasetMetadata`."""

    doi = data.get("doi")
    version = data.get("version")
    if not doi or not version:
        raise ValueError("metadata requires 'doi' and 'version'")
    return DatasetMetadata(doi=str(doi), version=str(version))


def _load(path: Path | str) -> DatasetMetadata:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = _require_metadata_fields(data)
    logger.info("Loaded chemistry metadata: doi=%s version=%s", meta.doi, meta.version)
    return meta


def load_adas_metadata(meta_path: Path | str) -> DatasetMetadata:
    """Load metadata for an ADAS table."""

    return _load(meta_path)


def load_lxcat_metadata(meta_path: Path | str) -> DatasetMetadata:
    """Load metadata for an LXCat cross-section table."""

    return _load(meta_path)


__all__ = ["DatasetMetadata", "load_adas_metadata", "load_lxcat_metadata"]
