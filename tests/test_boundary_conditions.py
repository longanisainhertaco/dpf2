import json
import pytest

from boundary_conditions import BoundaryConditions, BoundaryTypeEnum


def test_2d_rz_requires_reflecting_y():
    data = BoundaryConditions.with_defaults().model_dump(by_alias=True)
    data["yLow"] = "absorbing"
    with pytest.raises(ValueError):
        BoundaryConditions.model_validate(data, context={"geometry": "2D_RZ"})


def test_invalid_face_in_override_raises():
    data = BoundaryConditions.with_defaults().model_dump(by_alias=True)
    data["boundaryFieldOverrides"] = {"E": {"fakeFace": "reflecting"}}
    with pytest.raises(ValueError):
        BoundaryConditions.model_validate(data)


def test_periodic_faces_must_be_paired():
    data = BoundaryConditions.with_defaults().model_dump(by_alias=True)
    data["xLow"] = "periodic"
    with pytest.raises(ValueError):
        BoundaryConditions.model_validate(data)


def test_excluded_face_blocks_field_setting():
    data = BoundaryConditions.with_defaults().model_dump(by_alias=True)
    data["excludedFaces"] = ["zHigh"]
    with pytest.raises(ValueError):
        BoundaryConditions.model_validate(data)


def test_conflict_resolution_policy_respected():
    data = BoundaryConditions.with_defaults().model_dump(by_alias=True)
    data["yLow"] = "absorbing"
    data["conflictResolutionPolicy"] = "prefer_geometry"
    cfg = BoundaryConditions.model_validate(data, context={"geometry": "2D_RZ"})
    assert cfg.y_low is BoundaryTypeEnum.REFLECTING


def test_round_trip_yaml():
    cfg = BoundaryConditions.with_defaults()
    dumped = json.loads(json.dumps(cfg.model_dump(by_alias=True)))
    loaded = BoundaryConditions.model_validate(dumped)
    object.__setattr__(cfg, "_context", {})
    assert loaded == cfg
