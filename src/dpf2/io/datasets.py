"""Utilities for working with reference data sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_dataset_manifest() -> Dict[str, Any]:
    """Load the reference dataset manifest.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON description of available reference shots.
    """
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = repo_root / "Reference" / "dataset_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


__all__ = ["load_dataset_manifest"]
