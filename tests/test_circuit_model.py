import sys
from pathlib import Path

import importlib.util
from pathlib import Path
import sys
import types
import pytest

# Load CircuitModel without importing the full dpf2 package
pkg = types.ModuleType("dpf2")
pkg.__path__ = []
sim_pkg = types.ModuleType("dpf2.simulation")
sim_pkg.__path__ = []
sys.modules.setdefault("dpf2", pkg)
sys.modules.setdefault("dpf2.simulation", sim_pkg)
utils_stub = types.ModuleType("dpf2.simulation.utils")
class SimulationState:  # pragma: no cover - minimal stub
    """Placeholder simulation state."""
class FieldManager:  # pragma: no cover - minimal stub
    """Placeholder field manager."""
utils_stub.SimulationState = SimulationState
utils_stub.FieldManager = FieldManager
sys.modules["dpf2.simulation.utils"] = utils_stub
const_stub = types.ModuleType("dpf2.simulation.constants")
const_stub.e = const_stub.me = const_stub.epsilon0 = 1.0
sys.modules["dpf2.simulation.constants"] = const_stub
module_path = Path(__file__).resolve().parent.parent / "src/dpf2/simulation/circuit.py"
spec = importlib.util.spec_from_file_location("dpf2.simulation.circuit", module_path)
circuit_mod = importlib.util.module_from_spec(spec)
sys.modules["dpf2.simulation.circuit"] = circuit_mod
spec.loader.exec_module(circuit_mod)
CircuitModel = circuit_mod.CircuitModel


class DummyCollision:
    def spitzer_resistivity(self, *args, **kwargs):
        return 0.0


class DummyFieldManager:
    def get_J(self):
        """Return a zero current density for testing."""
        return 0.0


def test_negative_parameters_raise():
    with pytest.raises(ValueError, match="non-negative"):
        CircuitModel(
            C=-1.0,
            L0=1.0,
            R0=1.0,
            anode_radius=0.01,
            cathode_radius=0.02,
            collision_model=DummyCollision(),
            field_manager=DummyFieldManager(),
        )


def test_anode_radius_must_be_smaller():
    with pytest.raises(ValueError, match="Anode radius must be smaller"):
        CircuitModel(
            C=1.0,
            L0=1.0,
            R0=1.0,
            anode_radius=0.02,
            cathode_radius=0.01,
            collision_model=DummyCollision(),
            field_manager=DummyFieldManager(),
        )


def test_collision_model_requires_spitzer():
    class BadCollision:
        """Collision model lacking the required spitzer_resistivity."""

    with pytest.raises(ValueError, match="spitzer_resistivity"):
        CircuitModel(
            C=1.0,
            L0=1.0,
            R0=1.0,
            anode_radius=0.01,
            cathode_radius=0.02,
            collision_model=BadCollision(),
            field_manager=DummyFieldManager(),
        )
