import warnings
from pathlib import Path

import pytest

from warpx_settings import WarpXSettings, SpeciesEntry

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def make_base():
    data = WarpXSettings.with_defaults("3D_Cartesian").model_dump(by_alias=True)
    data["speciesConfig"] = {
        "e-": {
            "mass": 9.11e-31,
            "charge": -1.6e-19,
            "injection": "plasma",
        }
    }
    return data


def test_species_config_requires_mass_and_charge():
    data = WarpXSettings.with_defaults("3D_Cartesian").model_dump(by_alias=True)
    data["speciesConfig"] = {"e-": {"charge": -1.6e-19, "injection": "plasma"}}
    with pytest.raises(Exception):
        WarpXSettings.model_validate(data, context={"geometry": "3D_Cartesian"})


def test_invalid_boundary_keys_warn():
    data = make_base()
    data["boundaryConditions"] = {
        "e-": {
            "xLow": "reflecting",
            "xHigh": "absorbing",
            "yLow": "reflecting",
            "yHigh": "absorbing",
            "zLow": "reflecting",
            "zHigh": "absorbing",
            "bad": "periodic",
        }
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        WarpXSettings.model_validate(data, context={"geometry": "3D_Cartesian"})
        assert any("unrecognized" in str(wi.message) for wi in w)


def test_adaptive_step_requires_config():
    data = make_base()
    data["timeStepType"] = "adaptive"
    data.pop("adaptiveTimeStepConfig", None)
    with pytest.raises(ValueError):
        WarpXSettings.model_validate(data, context={"geometry": "3D_Cartesian"})


def test_yaml_round_trip_and_normalization(tmp_path: Path):
    cfg = WarpXSettings.with_defaults("2D_RZ")
    cfg = WarpXSettings.model_validate(cfg.model_dump(by_alias=True), context={"geometry": "2D_RZ"})
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return
    yaml_path = tmp_path / "w.yml"
    yaml.safe_dump({"warpx": cfg.model_dump(by_alias=True)}, open(yaml_path, "w"))
    loaded = yaml.safe_load(open(yaml_path))
    cfg2 = WarpXSettings.model_validate(loaded["warpx"], context={"geometry": "2D_RZ"})
    assert cfg2 == cfg


def test_summary_output():
    data = make_base()
    data.update({
        "fieldSolver": "PSATD",
        "particleShape": "quadratic",
        "particleShapeOrder": 2,
        "timeStepType": "adaptive",
        "adaptiveTimeStepConfig": {"cfl": 0.8, "dtMin": 1e-12, "dtMax": 1e-9},
        "ionizationModel": "ADK",
        "collisionModel": "MonteCarlo",
        "maxParticlesPerCell": 64,
        "currentSmoothingEnabled": True,
        "currentSmoothingKernel": [3, 3, 1],
        "emissionProfilePath": "profiles/emission_adk.csv",
    })
    cfg = WarpXSettings.model_validate(data, context={"geometry": "3D_Cartesian"})
    summary = cfg.summarize()
    assert "PSATD solver" in summary
    assert "Adaptive timestep" in summary
    assert "kernel=[3, 3, 1]" in summary


def test_hash_stability():
    d1 = make_base()
    cfg1 = WarpXSettings.model_validate(d1, context={"geometry": "3D_Cartesian"})
    d2 = make_base()
    cfg2 = WarpXSettings.model_validate(d2, context={"geometry": "3D_Cartesian"})
    assert cfg1.hash_warpx_config() == cfg2.hash_warpx_config()
