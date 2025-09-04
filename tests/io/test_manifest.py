from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import h5py_stub as h5py
sys.modules["h5py"] = h5py

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
    assert Path(meta["path"]) == data_file

    h5_path = tmp_path / "run_manifest.h5"
    with h5py.File(h5_path, "r") as h5:
        grp = h5["manifest/datasets/ADAS"]
        assert grp.attrs["doi"] == "10.1234/example"
        assert grp.attrs["version"] == "1.0"
        assert len(grp.attrs["hash"]) == 64


def test_manifest_dataset_requires_doi_and_version(tmp_path):
    data_file = tmp_path / "table.dat"
    data_file.write_text("x")
    with pytest.raises(ValueError):
        write_manifest(tmp_path, datasets={"ADAS": {"path": data_file, "doi": "10"}})
