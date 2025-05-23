import json
from pathlib import Path
import pytest

from radiation_transport import RadiationTransport

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def test_multigroup_requires_freq_edges():
    data = RadiationTransport.with_defaults().model_dump(by_alias=True)
    data["opticalDepthModel"] = "multi-group"
    with pytest.raises(ValueError):
        RadiationTransport.model_validate(data)


def test_closure_none_if_transport_off():
    data = RadiationTransport.with_defaults().model_dump(by_alias=True)
    data["transportModel"] = "none"
    data["closureMethod"] = "Eddington"
    with pytest.raises(ValueError):
        RadiationTransport.model_validate(data)


def test_opacity_table_path_required_if_tabulated(tmp_path: Path):
    data = RadiationTransport.with_defaults().model_dump(by_alias=True)
    data["opacitySource"] = "tabulated"
    with pytest.raises(ValueError):
        RadiationTransport.model_validate(data)


def test_camera_position_requires_raytrace_mode():
    data = RadiationTransport.with_defaults().model_dump(by_alias=True)
    data["raytraceCameraPosition"] = [0.0, 0.0, 1.0]
    data["transportModel"] = "FLD"
    with pytest.raises(ValueError):
        RadiationTransport.model_validate(data)


def test_yaml_round_trip_and_hash_stability(tmp_path: Path):
    cfg = RadiationTransport.model_validate(RadiationTransport.with_defaults().model_dump(by_alias=True))
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return
    p = tmp_path / "r.yml"
    yaml.safe_dump({"radiationTransport": cfg.model_dump(by_alias=True)}, open(p, "w"))
    loaded = yaml.safe_load(open(p))
    cfg2 = RadiationTransport.model_validate(loaded["radiationTransport"])
    assert cfg == cfg2
    assert cfg.radiation_transport_config_hash == cfg2.radiation_transport_config_hash


def test_species_list_validation_and_summary():
    data = RadiationTransport.with_defaults().model_dump(by_alias=True)
    data["opacitySpecies"] = ["D", "D"]
    with pytest.raises(ValueError):
        RadiationTransport.model_validate(data)

    cfg = RadiationTransport.with_defaults()
    summary = cfg.summarize()
    assert "D" in summary and "Ar" in summary
