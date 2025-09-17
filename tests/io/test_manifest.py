from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import h5py_stub as h5py

sys.modules["h5py"] = h5py

from dpf2.cli.lab import write_manifest


def test_manifest_records_dataset_doi(tmp_path):
    atomic = tmp_path / "atomic.dat"
    atomic.write_text("a")
    nuclear = tmp_path / "nuclear.dat"
    nuclear.write_text("n")
    material = tmp_path / "material.dat"
    material.write_text("m")
    manifest_path = write_manifest(
        tmp_path,
        datasets={
            "atomic": {
                "ADAS": {
                    "path": atomic,
                    "doi": "10.1234/example",
                    "version": "1.0",
                }
            },
            "nuclear": {
                "ENDF": {
                    "path": nuclear,
                    "doi": "10.5678/endf",
                    "version": "viii",
                }
            },
            "material": {
                "MatDB": {
                    "path": material,
                    "doi": "10.9999/matdb",
                    "version": "1.0",
                }
            },
        },
    )
    manifest = json.loads(manifest_path.read_text())
    assert "datasets" in manifest
    assert manifest["datasets"]["atomic"]["ADAS"]["doi"] == "10.1234/example"
    assert manifest["datasets"]["nuclear"]["ENDF"]["version"] == "viii"
    assert len(manifest["datasets"]["material"]["MatDB"]["hash"]) == 64

    h5_path = tmp_path / "run_manifest.h5"
    with h5py.File(h5_path, "r") as h5:
        grp = h5["manifest/datasets/atomic/ADAS"]
        assert grp.attrs["doi"] == "10.1234/example"
        grp = h5["manifest/datasets/nuclear/ENDF"]
        assert grp.attrs["version"] == "viii"
        grp = h5["manifest/datasets/material/MatDB"]
        assert len(grp.attrs["hash"]) == 64


def test_manifest_dataset_requires_doi_and_version(tmp_path):
    data_file = tmp_path / "table.dat"
    data_file.write_text("x")
    with pytest.raises(ValueError):
        write_manifest(
            tmp_path,
            datasets={"atomic": {"ADAS": {"path": data_file, "doi": "10"}}},
        )
