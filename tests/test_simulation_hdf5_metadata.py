import json
import hashlib
import dataclasses

import h5py_stub  # registers stub as h5py
from dpf2.core.config import DPFConfig
from dpf2.core.simulation import DPFSimulation
import h5py  # type: ignore


def test_hdf5_contains_metadata(tmp_path):
    cfg = DPFConfig()
    sim = DPFSimulation(cfg)
    seeds = {"python": 1, "numpy": 2}
    sim.run(end_time=0.0, output_dir=tmp_path, seeds=seeds)
    with h5py.File(tmp_path / "data_0.000000e+00.h5", "r") as f:
        meta_raw = f["metadata"][()]
        if hasattr(meta_raw, "data"):
            meta_raw = meta_raw.data
        if hasattr(meta_raw, "item"):
            meta_raw = meta_raw.item()
        meta = json.loads(meta_raw)
    expected = hashlib.sha256(
        json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode()
    ).hexdigest()
    assert meta["config_hash"] == expected
    assert meta["config"] == dataclasses.asdict(cfg)
    assert meta["seeds"] == seeds
    assert "git_commit" in meta
