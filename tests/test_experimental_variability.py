import pytest
from pathlib import Path

from dpf2.experimental_variability import ExperimentalVariabilityModel
from dpf2.experimental_variability import MonteCarloVariability
from dpf2.dpf_config import DPFConfig
import numpy as np


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


def test_monte_carlo_sampling(tmp_path):
    base_cfg = DPFConfig.with_defaults()
    var_cfg = ExperimentalVariabilityModel.with_defaults().model_copy(
        update={
            "pressure_jitter_pct": 5.0,
            "stochastic_run_id": 123,
            "per_field_distribution_params": {
                "capacitor_voltage": {"jitter_pct": 1.0},
                "cathode_gap_degrees": {"jitter_pct": 0.5},
            },
        }
    )
    sampler1 = MonteCarloVariability(var_cfg, realizations=4)
    sampler2 = MonteCarloVariability(var_cfg, realizations=4)
    v1 = sampler1.sample_capacitor_voltage(20e3)
    v2 = sampler2.sample_capacitor_voltage(20e3)
    assert np.allclose(v1, v2)
    pressures = sampler1.sample_fill_pressure(10.0)
    assert pressures.size == 4
    geom = sampler1.sample_geometry_tolerances({"cathode_gap_degrees": 36.0})
    assert len(geom) == 4 and "cathode_gap_degrees" in geom[0]
