import json
import json
import pytest

from dpf2.core_schema import DPFConfig, GeometryType, ModeType, MaterialOpacity
from dpf2.simulation_settings import SimulationSettings
from dpf2.grid_resolution import GridResolution


def test_defaults_create_section_models():
    cfg = DPFConfig.with_defaults()
    assert isinstance(cfg.simulation, SimulationSettings)
    assert cfg.simulation.geometry == GeometryType.RZ_2D
    assert cfg.grid.ny == 1


def test_simulation_time_validation():
    with pytest.raises(ValueError):
        SimulationSettings(time_start=1.0, time_end=0.0)


def test_grid_dimension_validation():
    with pytest.raises(ValueError):
        GridResolution.model_validate(
            {
                "nx": 0,
                "ny": 1,
                "nz": 1,
                "xMin": 0.0,
                "xMax": 1.0,
                "yMin": 0.0,
                "yMax": 1.0,
                "zMin": 0.0,
                "zMax": 1.0,
            }
        )


def test_pic_mode_requires_neutrals():
    cfg = DPFConfig.with_defaults()
    data = cfg.model_dump()
    data["simulation"]["mode"] = ModeType.PIC
    with pytest.raises(ValueError):
        DPFConfig.model_validate(data)


def test_config_round_trip():
    cfg = DPFConfig.with_defaults()
    json_str = cfg.model_dump_json(by_alias=True)
    data = json.loads(json_str)
    assert "simulation" in data
    reloaded = DPFConfig.model_validate(data)
    # Reloaded configuration should match the original instance
    assert reloaded.model_dump() == cfg.model_dump()
    # Serializing again should reproduce the original JSON representation
    assert json.loads(reloaded.model_dump_json(by_alias=True)) == data


def test_invalid_geometry():
    cfg = DPFConfig.with_defaults()
    data = cfg.model_dump()
    data["grid"]["ny"] = 2
    with pytest.raises(ValueError):
        DPFConfig.model_validate(data)


def test_required_fields():
    cfg = DPFConfig.with_defaults()
    assert "created_at" in cfg.required_fields()


def test_material_opacity_serialization_round_trip():
    cfg = DPFConfig.with_defaults()
    rad = cfg.radiation.model_copy(
        update={
            "group_count": 2,
            "group_opacities": [0.1, 0.2],
            "material_opacities": [
                MaterialOpacity(material_id="mat1", group_opacities=[1.0, 2.0])
            ],
        }
    )
    cfg = cfg.model_copy(update={"radiation": rad})

    dumped = cfg.model_dump()
    loaded = DPFConfig.model_validate(dumped)
    assert loaded.radiation.material_opacities[0].material_id == "mat1"
    assert loaded.radiation.material_opacities[0].group_opacities == [1.0, 2.0]
