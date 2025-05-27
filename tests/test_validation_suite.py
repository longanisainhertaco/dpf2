import json
from pathlib import Path

import pytest

from validation_suite import ValidationSuite

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def base_data(tmp_path: Path):
    d = tmp_path / "exp"
    d.mkdir(exist_ok=True)
    f1 = d / "cur.csv"
    f1.write_text("t,I\n0,0")
    f2 = d / "y.csv"
    f2.write_text("i,y\n0,1")
    return {
        "experimentDeviceId": "PF1000",
        "experimentCampaignId": "shot1",
        "datasetDirectory": d,
        "datasetFormat": "csv",
        "observableFileMap": {"I(t)": f1, "Yn": f2},
        "observableFormatSpec": {"I(t)": {"time": "t", "value": "I"}, "Yn": {"time": "i", "value": "y"}},
        "validationTargets": ["I(t)", "Yn"],
        "observableTolerances": {"I(t)": 0.1, "Yn": 0.2},
    }


def test_all_paths_must_exist(tmp_path: Path):
    data = base_data(tmp_path)
    data["datasetDirectory"] = tmp_path / "missing"
    with pytest.raises(ValueError):
        ValidationSuite.model_validate(data)
    data = base_data(tmp_path)
    data["observableFileMap"]["Yn"] = tmp_path / "no.csv"
    with pytest.raises(ValueError):
        ValidationSuite.model_validate(data)


def test_format_spec_requires_time_and_value(tmp_path: Path):
    data = base_data(tmp_path)
    del data["observableFormatSpec"]["I(t)"]["time"]
    with pytest.raises(ValueError):
        ValidationSuite.model_validate(data)


def test_yaml_round_trip_and_summary_output(tmp_path: Path):
    data = base_data(tmp_path)
    yaml_text = None
    if YAML_AVAILABLE:
        yaml_text = yaml.safe_dump({"validationSuite": data})
        p = tmp_path / "v.yml"
        p.write_text(yaml_text)
        loaded = yaml.safe_load(p.read_text())
        cfg = ValidationSuite.model_validate(loaded["validationSuite"])
        dumped = json.loads(cfg.model_dump_json(by_alias=True))
        cfg2 = ValidationSuite.model_validate(dumped)
        assert cfg == cfg2
        summary = cfg.summarize()
        assert "Validation Suite" in summary
    else:
        with pytest.raises(Exception):
            __import__("yaml")


def test_missing_target_handling_when_not_required(tmp_path: Path):
    data = base_data(tmp_path)
    data["requireAllTargets"] = False
    data["observableFileMap"].pop("Yn")
    data["observableFormatSpec"].pop("Yn")
    cfg = ValidationSuite.model_validate(data)
    assert "Yn" not in cfg.observable_file_map


def test_uncertainty_and_weighting_combined_score(tmp_path: Path):
    data = base_data(tmp_path)
    data["observableUncertainties"] = {"I(t)": 0.05, "Yn": 0.1}
    data["observableWeighting"] = {"I(t)": 0.7, "Yn": 0.3}
    data["validationScoreModel"] = "weighted"
    cfg = ValidationSuite.model_validate(data)
    assert cfg.computed_validation_score == pytest.approx(0.935)
    assert cfg.validation_passed


def test_hash_changes_with_score_model_or_dataset(tmp_path: Path):
    data = base_data(tmp_path)
    cfg1 = ValidationSuite.model_validate(data)
    cfg2 = ValidationSuite.model_validate(data)
    assert cfg1.hash_validation_suite_config() == cfg2.hash_validation_suite_config()
    data["validationScoreModel"] = "MAE"
    cfg3 = ValidationSuite.model_validate(data)
    assert cfg1.hash_validation_suite_config() != cfg3.hash_validation_suite_config()
