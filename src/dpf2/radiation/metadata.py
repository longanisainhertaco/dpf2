from __future__ import annotations

"""Metadata loaders for radiation data sets."""

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
    doi = data.get("doi")
    version = data.get("version")
    if not doi or not version:
        raise ValueError("metadata requires 'doi' and 'version'")
    return DatasetMetadata(doi=str(doi), version=str(version))


def load_chianti_metadata(meta_path: Path | str) -> DatasetMetadata:
    """Load metadata for a CHIANTI table."""

    data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    meta = _require_metadata_fields(data)
    logger.info(
        "Loaded CHIANTI metadata: doi=%s version=%s", meta.doi, meta.version
    )
    return meta


__all__ = ["DatasetMetadata", "load_chianti_metadata"]
