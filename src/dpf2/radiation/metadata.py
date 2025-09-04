from __future__ import annotations

"""Metadata loaders for radiation data sets.

These helpers validate that required provenance information is present in
sidecar JSON metadata files distributed with CHIANTI or other radiation
tables.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
import logging


@dataclass
class DatasetMetadata:
    """Minimal provenance for a radiation table."""

    doi: str
    version: str


logger = logging.getLogger(__name__)


def _require_metadata_fields(data: Mapping[str, object]) -> DatasetMetadata:
    """Validate DOI and version fields and build :class:`DatasetMetadata`."""

    doi = data.get("doi")
    version = data.get("version")
    if not doi or not version:
        raise ValueError("metadata requires 'doi' and 'version'")
    return DatasetMetadata(doi=str(doi), version=str(version))


def _load(path: Path | str) -> DatasetMetadata:
    """Load and validate a metadata JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = _require_metadata_fields(data)
    logger.info("Loaded radiation metadata: doi=%s version=%s", meta.doi, meta.version)
    return meta


def load_chianti_metadata(meta_path: Path | str) -> DatasetMetadata:
    """Load metadata for a CHIANTI table."""

    return _load(meta_path)


__all__ = ["DatasetMetadata", "load_chianti_metadata"]
