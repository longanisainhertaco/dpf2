import numpy as np
import pytest
from scipy.constants import mu_0

from dpf2.simulation.circuit import CircuitModel
from dpf2.simulation.utils import FieldManager


class DummyCollisionModel:
    def spitzer_resistivity(self, ne, Te, lnL):
        return 1.0  # Constant resistivity for testing


def make_circuit(a=0.01, b=0.05):
    field_manager = FieldManager((1, 1, 1), 1.0, 1.0, 1.0, (0, 0, 0), {})
    cm = DummyCollisionModel()
    return CircuitModel(
        C=1e-6,
        L0=1e-7,
        R0=0.1,
        anode_radius=a,
        cathode_radius=b,
        collision_model=cm,
        field_manager=field_manager,
    )


def test_plasma_inductance_matches_reference():
    circuit = make_circuit()
    z = 0.02
    state = type("State", (), {"sheath_position": z})()
    L_model = circuit.plasma_inductance(state)
    L_ref = mu_0 / (2 * np.pi) * z * np.log(0.05 / 0.01)
    assert np.isclose(L_model, L_ref)


def test_plasma_inductance_invalid_state():
    circuit = make_circuit()
    state = type("State", (), {"sheath_position": -0.1})()
    with pytest.raises(ValueError):
        circuit.plasma_inductance(state)
