import json

from dpf2.cli.lab import write_manifest


def test_manifest_includes_config_and_seeds(tmp_path):
    cfg = {"foo": 1}
    seeds = {"python": 42, "numpy": 99}
    path = write_manifest(tmp_path, config=cfg, seeds=seeds)
    data = json.loads(path.read_text())
    assert data["code_hash"]
    assert data["config"] == cfg
    assert data["random_seeds"] == seeds
    assert "environment" in data and "python" in data["environment"]
