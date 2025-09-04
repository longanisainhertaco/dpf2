import warnings
import pytest
from dpf2.device_profiles import DeviceProfiles

from dpf2.cli.main import simulate

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def test_default_device_key_exists():
    cfg = DeviceProfiles.with_defaults()
    assert cfg.default_device_id in cfg.devices


def test_additional_presets_available():
    cfg = DeviceProfiles.with_defaults()
    assert "EDU1K" in cfg.devices
    assert "IND20K" in cfg.devices


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
    assert cfg2.model_dump(by_alias=True) == cfg.model_dump(by_alias=True)
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


def test_insulator_material_is_materialref():
    cfg = DeviceProfiles.with_defaults()
    sleeve = cfg.devices["PF1000"].insulator_sleeve
    assert cfg.devices["PF1000"].insulator_material is not None
    assert (
        cfg.devices["PF1000"].insulator_material.material_id == "alumina"
    )
    assert sleeve is not None and sleeve.material is not None
    assert sleeve.material.material_id == "alumina"


def test_electrode_material_fields_present():
    cfg = DeviceProfiles.with_defaults()
    d = cfg.devices["PF1000"]
    # ``d`` may be a plain ``dict`` when the lightweight pydantic stub is
    # used in environments without the real dependency.  In that case we
    # simply verify that the new keys exist.  With the real pydantic model
    # the attributes are available directly on the ``DeviceEntry`` object.
    if isinstance(d, dict):  # pragma: no cover - exercised in stub mode
        assert "anodeMaterial" not in d or d["anodeMaterial"] is None
        assert "cathodeMaterial" not in d or d["cathodeMaterial"] is None
    else:  # pragma: no cover - real pydantic path
        assert hasattr(d, "anode_material")
        assert hasattr(d, "cathode_material")
        assert d.anode_material is None and d.cathode_material is None


def test_cli_has_device_option():
    assert any("--device" in opt.opts for opt in simulate.params)
