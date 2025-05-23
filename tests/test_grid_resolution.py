import json
import warnings

import pytest

from grid_resolution import GridResolution


def test_geometry_2d_rz_flat_y():
    cfg = GridResolution.with_defaults("2D_RZ")
    data = cfg.model_dump(by_alias=True)
    data["yMax"] = 1.0
    with pytest.raises(ValueError):
        GridResolution.model_validate(data, context={"geometry": "2D_RZ"})


def test_domain_extent_ordering_enforced():
    data = GridResolution.with_defaults("3D_Cartesian").model_dump(by_alias=True)
    data["xMax"] = data["xMin"]
    with pytest.raises(ValueError):
        GridResolution.model_validate(data, context={"geometry": "3D_Cartesian"})


def test_padding_cells_validation():
    data = GridResolution.with_defaults("3D_Cartesian").model_dump(by_alias=True)
    data["paddingCells"] = {"fake": -1}
    with pytest.raises(ValueError):
        GridResolution.model_validate(data, context={"geometry": "3D_Cartesian"})


def test_aspect_ratio_warning():
    data = GridResolution.with_defaults("3D_Cartesian").model_dump(by_alias=True)
    data["xMax"] = 50.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        GridResolution.model_validate(data, context={"geometry": "3D_Cartesian"})
        assert any("aspect ratio" in str(wi.message) for wi in w)


def test_hash_stability_and_summary():
    cfg1 = GridResolution.with_defaults("2D_RZ")
    cfg1 = GridResolution.model_validate(cfg1.model_dump(by_alias=True), context={"geometry": "2D_RZ"})
    cfg2 = GridResolution.model_validate(cfg1.model_dump(by_alias=True), context={"geometry": "2D_RZ"})
    assert cfg1.grid_config_hash == cfg2.grid_config_hash
    summary = cfg1.summarize()
    assert cfg1.grid_config_hash[:6] in summary


def test_nonuniform_axis_scaling_validation():
    data = GridResolution.with_defaults("3D_Cartesian").model_dump(by_alias=True)
    data["nonuniformAxisScaling"] = {"a": "log"}
    with pytest.raises(ValueError):
        GridResolution.model_validate(data, context={"geometry": "3D_Cartesian"})


def test_normalize_units_applies_correct_scale():
    cfg = GridResolution.with_defaults("2D_RZ")
    scaled = cfg.normalize_units("m")
    assert scaled.x_max == pytest.approx(cfg.x_max * 0.01)


def test_grid_centering_affects_odd_even_logic():
    data = GridResolution.with_defaults("3D_Cartesian").model_dump(by_alias=True)
    data["gridCentering"] = "node"
    data["nx"] = 10
    data["nz"] = 12
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        GridResolution.model_validate(data, context={"geometry": "3D_Cartesian"})
        assert any("odd" in str(wi.message) for wi in w)

