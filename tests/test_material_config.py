import pytest

from dpf2.simulation.config_schema import MaterialConfig
from dpf2.materials.models import MaterialRef


def test_component_materials_use_materialref():
    cfg = MaterialConfig(components={"anode": MaterialRef(material_id="copper")})
    assert isinstance(cfg.components["anode"], MaterialRef)
    assert cfg.components["anode"].material_id == "copper"


def test_unknown_material_validation():
    with pytest.raises(ValueError):
        MaterialConfig._validate_material_ids(MaterialConfig, {"anode": MaterialRef(material_id="bad")})

