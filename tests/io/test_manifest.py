from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpf2.cli.lab import write_manifest


def test_manifest_records_dataset_doi(tmp_path):
    data_file = tmp_path / "table.dat"
    data_file.write_text("example")
    manifest_path = write_manifest(
        tmp_path,
        datasets={
            "ADAS": {
                "path": data_file,
                "doi": "10.1234/example",
                "version": "1.0",
            }
        },
    )
    manifest = json.loads(manifest_path.read_text())
    assert "datasets" in manifest
    meta = manifest["datasets"]["ADAS"]
    assert meta["doi"] == "10.1234/example"
    assert meta["version"] == "1.0"
    assert len(meta["hash"]) == 64


def test_manifest_dataset_requires_doi_and_version(tmp_path):
    data_file = tmp_path / "table.dat"
    data_file.write_text("x")
    with pytest.raises(ValueError):
        write_manifest(tmp_path, datasets={"ADAS": {"path": data_file, "doi": "10"}})
