import json

from dpf2.cli.lab import write_manifest


def test_manifest_includes_config_and_seeds(tmp_path, monkeypatch):
    cfg = {"foo": 1}
    seeds = {"python": 42, "numpy": 99}
    monkeypatch.setenv("CONTAINER_HASH", "sha256:abc")
    monkeypatch.setenv("CC", "gcc")
    monkeypatch.setenv("MPI_VERSION", "OpenMPI 4.1")
    path = write_manifest(tmp_path, config=cfg, seeds=seeds)
    data = json.loads(path.read_text())
    assert data["code_hash"]
    assert data["config"] == cfg
    assert data["random_seeds"] == seeds
    assert "environment" in data
    env = data["environment"]
    assert env["container_hash"] == "sha256:abc"
    assert env["compiler"] == "gcc"
    assert env["mpi"] == "OpenMPI 4.1"
    assert isinstance(env["hdf5"], str)
    assert env["python"]
