import pytest

from dpf2.materials.library import MaterialLibrary
from dpf2.materials.state import ComponentMaterialState
from dpf2.materials.mdm import MaterialDamageModel
from dpf2.materials import MaterialRef


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
    assert state.redeposited_mass == pytest.approx(0.0)
    assert state.contamination_thickness == pytest.approx(0.0)


def test_state_serialization_roundtrip():
    mat = MaterialLibrary.get("copper")
    original = ComponentMaterialState(
        material=mat,
        erosion=1.0,
        redeposited_mass=0.1,
        contamination_thickness=0.2,
        temperature_history=[300.0, 310.0],
    )
    data = original.to_dict()
    restored = ComponentMaterialState.from_dict(data)
    assert restored.material == mat
    assert restored.erosion == pytest.approx(1.0)
    assert restored.redeposited_mass == pytest.approx(0.1)
    assert restored.contamination_thickness == pytest.approx(0.2)
    assert restored.temperature_history == [300.0, 310.0]



def test_redeposition_and_film():
    copper = MaterialLibrary.get("copper")
    steel = MaterialLibrary.get("stainless_steel")
    state_a = ComponentMaterialState(material=copper)
    state_b = ComponentMaterialState(material=steel)

    class DummySolver:
        def surface_flux(self, component: str) -> float:
            return 10.0 if component == "A" else 0.0

        def surface_temperature(self, component: str) -> float:
            return 400.0

        def deposition_flux(self, component: str) -> float:
            return 0.02 if component == "A" else 0.0

        def evaporation_rate(self, component: str, temp: float) -> float:
            return 0.01 if component == "A" else 0.0

    class DummyPlasma:
        def __init__(self):
            self.calls = []

        def inject_impurities(self, name: str, amount: float) -> None:
            self.calls.append((name, amount))

    plasma = DummyPlasma()
    mdm = MaterialDamageModel({"A": state_a, "B": state_b}, plasma_model=plasma)
    mdm.apply(DummySolver(), dt=1.0)

    sputter = 10.0 * copper.sputter_yield * 1.0
    redep = 0.02 * 1.0
    net = max(sputter - redep, 0.0)
    evap = 0.01 * 1.0

    assert state_a.erosion == pytest.approx(net + evap)
    assert state_a.redeposited_mass == pytest.approx(redep)
    assert state_b.contamination_thickness == pytest.approx(net)
    # Plasma model should receive negative redeposition and positive evaporation
    assert ("A", -redep) in plasma.calls
    assert ("A", evap) in plasma.calls

