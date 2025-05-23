import json
from pathlib import Path
import pytest

from initial_conditions import InitialConditions, BreakdownModel, PaschenModel


def test_missing_field_threshold_fails():
    with pytest.raises(ValueError):
        BreakdownModel(type="FN")


def test_invalid_preionization_logic():
    with pytest.raises(ValueError):
        InitialConditions(
            temperature=15000,
            density=1e-4,
            gas_type="D2",
            sheath_type="gaussian",
            sheath_velocity_profile=[(0.0, 1.0)],
            current_profile=[(0.0, 10.0)],
            preionization_method="UV",
            breakdown_model=BreakdownModel.with_defaults(),
            paschen_model=PaschenModel.with_defaults(),
            enable_dynamic_ionization_rate=True,
        )


def test_empty_current_profile_raises():
    with pytest.raises(ValueError):
        InitialConditions(
            temperature=15000,
            density=1e-4,
            gas_type="D2",
            sheath_type="slab",
            sheath_velocity_profile=[(0.0, 0.0)],
            current_profile=[],
            breakdown_model=BreakdownModel.with_defaults(),
            paschen_model=PaschenModel.with_defaults(),
            enable_dynamic_ionization_rate=False,
        )


def test_round_trip_serialization_yaml(tmp_path: Path):
    ic = InitialConditions.with_defaults()
    data = ic.model_dump()
    yaml_str = json.dumps(data)
    loaded = InitialConditions.model_validate(json.loads(yaml_str))
    assert loaded == ic
