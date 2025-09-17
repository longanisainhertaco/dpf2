import json
from pathlib import Path
import pytest

from dpf2.xray_emission_model import XrayEmissionModel

try:
    import yaml  # type: ignore

    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def test_energy_bins_are_monotonic():
    cfg = XrayEmissionModel.model_validate({"xray_energy_bins": [1.0, 0.5, 2.0]})
    with pytest.raises(ValueError):
        cfg._run_validations()


def test_missing_filter_path_raises(tmp_path: Path):
    cfg = XrayEmissionModel.model_validate(
        {
            "apply_detector_filter": True,
            "xray_detector_filter_path": tmp_path / "f.csv",
        }
    )
    with pytest.raises(ValueError):
        cfg._run_validations()


def test_custom_mask_requires_file(tmp_path: Path):
    cfg = XrayEmissionModel.model_validate(
        {
            "emission_volume_specification": "custom_mask",
            "custom_emission_mask_path": tmp_path / "mask.h5",
        }
    )
    with pytest.raises(ValueError):
        cfg._run_validations()


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
    cfg = XrayEmissionModel.model_validate(
        {
            "atomic_data_source": "NIST",
            "ion_species": ["Unknown"],
        }
    )
    with pytest.warns(UserWarning):
        cfg._run_validations()
