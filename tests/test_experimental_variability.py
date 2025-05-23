import pytest
from pathlib import Path

from experimental_variability import ExperimentalVariabilityModel


def test_invalid_jitter_values():
    data = ExperimentalVariabilityModel.with_defaults().model_dump()
    data["pressure_jitter_pct"] = 120.0
    with pytest.raises(ValueError):
        ExperimentalVariabilityModel.model_validate(data)
    data["pressure_jitter_pct"] = 10.0
    data["trigger_jitter_ns"] = -1.0
    with pytest.raises(ValueError):
        ExperimentalVariabilityModel.model_validate(data)
    data["trigger_jitter_ns"] = 1.0
    data["erosion_multiplier"] = 15.0
    with pytest.raises(ValueError):
        ExperimentalVariabilityModel.model_validate(data)


def test_conflicting_profile_policy(tmp_path: Path):
    path = tmp_path / "erosion.csv"
    path.write_text("data")
    data = ExperimentalVariabilityModel.with_defaults().model_dump()
    data.update({
        "erosion_multiplier": 1.2,
        "erosion_profile_from_file": path,
        "profile_conflict_policy": "error",
    })
    with pytest.raises(ValueError):
        ExperimentalVariabilityModel.model_validate(data)


def test_missing_profile_path_raises():
    data = ExperimentalVariabilityModel.with_defaults().model_dump()
    data.update({
        "time_varying_environment_model": "from_file",
        "time_varying_profile_path": None,
    })
    with pytest.raises(ValueError):
        ExperimentalVariabilityModel.model_validate(data)


def test_distribution_override_behavior():
    data = ExperimentalVariabilityModel.with_defaults().model_dump()
    data.update({
        "distribution_model": "uniform",
        "per_field_distributions": {"trigger_jitter_ns": "normal"},
    })
    cfg = ExperimentalVariabilityModel.model_validate(data)
    assert cfg.per_field_distributions["trigger_jitter_ns"] == "normal"


def test_config_hash_changes_on_seed():
    d1 = ExperimentalVariabilityModel.with_defaults().model_dump()
    d1["stochastic_run_id"] = 1
    cfg1 = ExperimentalVariabilityModel.model_validate(d1)

    d2 = ExperimentalVariabilityModel.with_defaults().model_dump()
    d2["stochastic_run_id"] = 2
    cfg2 = ExperimentalVariabilityModel.model_validate(d2)

    assert cfg1.variability_config_hash != cfg2.variability_config_hash
