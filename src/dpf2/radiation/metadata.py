from __future__ import annotations

"""Metadata loaders for radiation data sets."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class DatasetMetadata:
    """Minimal provenance for a radiation table."""

    doi: str
    version: str


def load_chianti_metadata(meta_path: Path | str) -> DatasetMetadata:
    """Load metadata for a CHIANTI table."""

    data = json.loads(Path(meta_path).read_text())
    doi = data.get("doi")
    version = data.get("version")
    if not doi or not version:
        raise ValueError("metadata requires 'doi' and 'version'")
    return DatasetMetadata(doi=str(doi), version=str(version))


__all__ = ["DatasetMetadata", "load_chianti_metadata"]
