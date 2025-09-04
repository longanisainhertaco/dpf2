import pytest

import pytest

from dpf2.materials import (
    MaterialLibrary,
    ComponentMaterialState,
    MaterialDamageModel,
)


def test_material_lookup():
    mat = MaterialLibrary.get("copper")
    assert mat.name == "copper"
    assert mat.density == pytest.approx(8960.0)


def test_erosion_accumulation():
    mat = MaterialLibrary.get("tungsten")
    state = ComponentMaterialState(material=mat)

    class DummySolver:
        def surface_flux(self, component: str) -> float:
            return 5.0

        def surface_temperature(self, component: str) -> float:
            return 350.0

    mdm = MaterialDamageModel({"anode": state})
    mdm.apply(DummySolver(), dt=2.0)

    expected = 5.0 * mat.sputter_yield * 2.0
    assert state.erosion == pytest.approx(expected)
    assert state.temperature_history[-1] == 350.0


def test_state_serialization_roundtrip():
    mat = MaterialLibrary.get("copper")
    original = ComponentMaterialState(
        material=mat,
        erosion=1.0,
        film_thickness=0.2,
        temperature_history=[300.0, 310.0],
    )
    data = original.to_dict()
    restored = ComponentMaterialState.from_dict(data)
    assert restored.material == mat
    assert restored.erosion == pytest.approx(1.0)
    assert restored.film_thickness == pytest.approx(0.2)
    assert restored.temperature_history == [300.0, 310.0]
