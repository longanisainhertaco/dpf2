import json
from pathlib import Path
import pytest

from xray_emission_model import XrayEmissionModel

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def test_energy_bins_are_monotonic():
    data = XrayEmissionModel.with_defaults().model_dump(by_alias=True)
    data["xrayEnergyBins"] = [1.0, 0.5, 2.0]
    with pytest.raises(ValueError):
        XrayEmissionModel.model_validate(data)


def test_missing_filter_path_raises(tmp_path: Path):
    data = XrayEmissionModel.with_defaults().model_dump(by_alias=True)
    data.update({"applyDetectorFilter": True, "xrayDetectorFilterPath": tmp_path / "f.csv"})
    with pytest.raises(ValueError):
        XrayEmissionModel.model_validate(data)


def test_custom_mask_requires_file(tmp_path: Path):
    data = XrayEmissionModel.with_defaults().model_dump(by_alias=True)
    data.update({"emissionVolumeSpecification": "custom_mask", "customEmissionMaskPath": tmp_path / "mask.h5"})
    with pytest.raises(ValueError):
        XrayEmissionModel.model_validate(data)


def test_yaml_round_trip_and_summary(tmp_path: Path):
    cfg = XrayEmissionModel.with_defaults()
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return
    p = tmp_path / "x.yml"
    yaml.safe_dump({"xrayEmission": cfg.model_dump(by_alias=True)}, open(p, "w"))
    loaded = yaml.safe_load(open(p))
    cfg2 = XrayEmissionModel.model_validate(loaded["xrayEmission"])
    assert cfg == cfg2
    assert "X-ray" in cfg.summarize()


def test_species_validated_if_noncustom_db():
    data = XrayEmissionModel.with_defaults().model_dump(by_alias=True)
    data.update({"atomicDataSource": "NIST", "ionSpecies": ["Unknown"]})
    with pytest.warns(UserWarning):
        XrayEmissionModel.model_validate(data)
