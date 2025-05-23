import json
from pathlib import Path
import pytest

from advanced_options import AdvancedOptions


def test_disable_validators_requires_reason():
    data = AdvancedOptions.with_defaults().model_dump(by_alias=True)
    data["disableAllValidators"] = True
    data["disableReason"] = None
    with pytest.raises(ValueError):
        AdvancedOptions.model_validate(data)


def test_runtime_script_path_must_exist(tmp_path: Path):
    data = AdvancedOptions.with_defaults().model_dump(by_alias=True)
    data["injectRuntimeScript"] = tmp_path / "missing.py"
    with pytest.raises(ValueError):
        AdvancedOptions.model_validate(data)


def test_tile_size_validator():
    data = AdvancedOptions.with_defaults().model_dump(by_alias=True)
    data["forceAmrexTileSize"] = [32, 0, 1]
    with pytest.raises(ValueError):
        AdvancedOptions.model_validate(data)


def test_summary_contains_hash_and_toggles(tmp_path: Path):
    script = tmp_path / "inject.py"
    script.write_text("pass")
    data = AdvancedOptions.with_defaults().model_dump(by_alias=True)
    data.update({
        "enableDiagnosticsMockMode": True,
        "forceAmrexTileSize": [32, 32, 1],
        "injectRuntimeScript": script,
        "amrexDebugLevel": 4,
    })
    cfg = AdvancedOptions.model_validate(data)
    summary = cfg.summarize()
    assert "diagnostics mocked = True" in summary
    assert "validators disabled = False" in summary
    assert "Hash:" in summary
    assert cfg.advanced_config_hash


def test_round_trip_serialization():
    cfg = AdvancedOptions.with_defaults()
    dumped = json.loads(cfg.model_dump_json(by_alias=True))
    loaded = AdvancedOptions.model_validate(dumped)
    assert loaded == cfg
