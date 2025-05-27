import warnings
import pytest
from device_profiles import DeviceProfiles

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def test_default_device_key_exists():
    cfg = DeviceProfiles.with_defaults()
    assert cfg.default_device_id in cfg.devices


def test_fuel_fractions_sum_to_one():
    data = DeviceProfiles.with_defaults().model_dump(by_alias=True)
    data["devices"]["PF1000"]["fuelMixture"] = {"D": 0.5, "Ar": 0.6}
    with pytest.raises(ValueError):
        DeviceProfiles.model_validate(data)


def test_radius_and_length_must_be_positive():
    data = DeviceProfiles.with_defaults().model_dump(by_alias=True)
    data["devices"]["PF1000"]["anodeRadiusCm"] = -1.0
    with pytest.raises(ValueError):
        DeviceProfiles.model_validate(data)


def test_yaml_round_trip_and_summary_output(tmp_path):
    cfg = DeviceProfiles.with_defaults()
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return
    yaml_path = tmp_path / "d.yml"
    yaml.safe_dump({"deviceProfiles": cfg.model_dump(by_alias=True)}, open(yaml_path, "w"))
    loaded = yaml.safe_load(open(yaml_path))
    cfg2 = DeviceProfiles.model_validate(loaded["deviceProfiles"])
    assert cfg2.devices["PF1000"].anode_radius_cm == cfg.devices["PF1000"].anode_radius_cm
    summary = cfg.summarize()
    assert "Devices:" in summary and "PF1000" in summary


def test_missing_bank_fields_warn():
    data = DeviceProfiles.with_defaults().model_dump(by_alias=True)
    data["devices"]["PF1000"]["capacitorBank"].pop("R")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DeviceProfiles.model_validate(data)
        assert any("capacitor bank" in str(wi.message) for wi in w)


def test_hash_changes_on_geometry_change():
    cfg = DeviceProfiles.with_defaults()
    base_hash = cfg.device_profiles_config_hash
    data = cfg.model_dump(by_alias=True)
    data["devices"]["PF1000"]["anodeLengthCm"] = cfg.devices["PF1000"].anode_length_cm * 2
    cfg2 = DeviceProfiles.model_validate(data)
    assert base_hash != cfg2.device_profiles_config_hash
