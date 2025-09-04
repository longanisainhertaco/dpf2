from pathlib import Path

from dpf2.core.config import DPFConfig


def test_roundtrip_json(tmp_path: Path):
    cfg = DPFConfig()
    path = tmp_path / "cfg.json"
    cfg.to_file(path)
    loaded = DPFConfig.from_file(path)
    assert loaded == cfg
