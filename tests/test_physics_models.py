import json
from pathlib import Path
import pytest

from physics_models import PhysicsModels
from core_schema import (
    EOSModel,
    ResistivityModel,
    IonizationModel,
    IonizationFallback,
    RadiationModel,
    RadiationTransportModel,
    LineEscapeMethod,
    RadiationGeometryModel,
    InstabilityModel,
)


def test_missing_bandpass_for_radiation():
    with pytest.raises(ValueError):
        PhysicsModels(
            eos_model=EOSModel.IDEAL,
            gamma=1.4,
            radiation_model=RadiationModel.LINE,
            radiation_transport_model=RadiationTransportModel.ESCAPE_FACTOR,
            line_escape_method=LineEscapeMethod.EDDINGTON,
            radiation_geometry_model=RadiationGeometryModel.SLAB,
        )


def test_ionization_fallback_triggered():
    data = {
        "eosModel": "ideal",
        "gamma": 1.4,
        "ionizationModel": "None",
        "fallbackIfIonizationInvalid": "switch_to_Saha",
    }
    cfg = PhysicsModels.model_validate(data, context={"gas_type": "D2"})
    assert cfg.fallback_if_ionization_invalid is IonizationFallback.SWITCH_TO_SAHA


def test_missing_thresholds_for_instabilities():
    with pytest.raises(ValueError):
        PhysicsModels(
            eos_model=EOSModel.IDEAL,
            gamma=1.4,
            instability_models_enabled=[InstabilityModel.KINK, InstabilityModel.SAUSAGE],
            instability_thresholds={"kink": 0.04},
        )


def test_valid_physics_round_trip():
    cfg = PhysicsModels.with_defaults()
    dumped = json.loads(cfg.model_dump_json(by_alias=True))
    loaded = PhysicsModels.model_validate(dumped)
    assert cfg == loaded


def test_opacity_required_for_montecarlo():
    with pytest.raises(ValueError):
        PhysicsModels(
            eos_model=EOSModel.IDEAL,
            gamma=1.4,
            radiation_model=RadiationModel.BREMSSTRAHLUNG,
            radiation_transport_model=RadiationTransportModel.MONTE_CARLO,
        )


def test_tabulated_eos_requires_table():
    with pytest.raises(ValueError):
        PhysicsModels(
            eos_model=EOSModel.TABULATED,
            gamma=1.4,
        )
