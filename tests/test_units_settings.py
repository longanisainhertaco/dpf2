import json
import pytest

from units_settings import UnitsSettings

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def test_unit_consistency_si_requires_m_and_ns():
    data = UnitsSettings.with_defaults().model_dump(by_alias=True)
    data["baseUnits"] = "SI"
    data["spatialUnits"] = "cm"
    with pytest.raises(ValueError):
        UnitsSettings.model_validate(data)


def test_invalid_override_raises():
    data = UnitsSettings.with_defaults().model_dump(by_alias=True)
    data["unitResolutionOverrides"] = {"invalid": 1.0}
    with pytest.raises(ValueError):
        UnitsSettings.model_validate(data)


def test_yaml_round_trip_and_hash_stability(tmp_path):
    cfg = UnitsSettings.with_defaults()
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return
    yaml_path = tmp_path / "u.yml"
    yaml.safe_dump({"units": cfg.model_dump(by_alias=True)}, open(yaml_path, "w"))
    loaded = yaml.safe_load(open(yaml_path))
    cfg2 = UnitsSettings.model_validate(loaded["units"])
    assert cfg.units_config_hash == cfg2.units_config_hash
    assert cfg == cfg2


def test_input_assumed_units_applied():
    data = UnitsSettings.with_defaults().model_dump(by_alias=True)
    data["inputAssumedUnits"] = {"rho": "g/cm3"}
    cfg = UnitsSettings.model_validate(data)
    unit_map = cfg.normalize_units()
    assert "rho" in unit_map


def test_output_unit_convertibility_check():
    data = UnitsSettings.with_defaults().model_dump(by_alias=True)
    data["preferredOutputUnits"] = {"E": "furlong"}
    with pytest.raises(ValueError):
        UnitsSettings.model_validate(data)

