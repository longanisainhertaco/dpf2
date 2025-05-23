import json
from pathlib import Path
import pytest

from dpf_config import DPFConfig

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def test_load_min_yaml(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_content = """
simulation_control:
  geometry: 2D_RZ
"""
    yaml_path.write_text(yaml_content)
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            DPFConfig.from_file(yaml_path)
    else:
        cfg = DPFConfig.from_file(yaml_path)
        assert cfg.simulation_control.geometry == "2D_RZ"


def test_roundtrip_json(tmp_path):
    cfg = DPFConfig.with_defaults()
    json_path = tmp_path / "cfg.json"
    cfg.to_json(json_path)
    cfg2 = DPFConfig.from_file(json_path)
    assert cfg2.simulation_control.mode == cfg.simulation_control.mode


def test_validation_failure():
    cfg = DPFConfig.with_defaults()
    data = cfg.model_dump()
    data["simulation_control"]["geometry"] = "2D_RZ"
    data["grid_resolution"]["ny"] = 5
    with pytest.raises(ValueError):
        DPFConfig.model_validate(data)


def test_summarize():
    cfg = DPFConfig.with_defaults()
    summary = cfg.summarize()
    assert "DPF Simulation Configuration Summary" in summary


def test_required_fields():
    cfg = DPFConfig.with_defaults()
    assert cfg.required_fields() == []
