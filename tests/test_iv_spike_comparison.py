import csv
import importlib.util
import types
from pathlib import Path
import sys
import numpy as np

# Load CircuitModel without importing the full dpf2 package
pkg = types.ModuleType("dpf2")
pkg.__path__ = []
sim_pkg = types.ModuleType("dpf2.simulation")
sim_pkg.__path__ = []
sys.modules.setdefault("dpf2", pkg)
sys.modules.setdefault("dpf2.simulation", sim_pkg)

utils_stub = types.ModuleType("dpf2.simulation.utils")


class SimulationState:
    """Minimal simulation state used for circuit tests."""

    def __init__(self):
        self.sheath_position = 0.1
        self.electron_temperature = 1e3
        self.density = 1e20


class FieldManager:
    """Placeholder field manager."""

    def get_J(self):  # pragma: no cover - not used
        return 0.0


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


def _load_spike_data():
    data_path = Path(__file__).parent / "data/iv_spike.csv"
    times, currents, voltages = [], [], []
    with data_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            currents.append(float(row["current"]))
            voltages.append(float(row["voltage"]))
    return times, currents, voltages


def test_iv_curve_matches_experiment():
    times, currents, voltages = _load_spike_data()

    class SimpleCircuit(CircuitModel):
        def plasma_inductance(self, state):  # pragma: no cover - simple override
            return 0.0

        def plasma_resistance(self, state):  # pragma: no cover - simple override
            return 0.0

    circuit = SimpleCircuit(
        C=1e-6,
        L0=1e-6,
        R0=0.0,
        anode_radius=0.01,
        cathode_radius=0.02,
        collision_model=DummyCollision(),
        field_manager=FieldManager(),
        V0=1000.0,
        switch_initial_resistance=1e-3,
        switch_final_resistance=1e-3,
        switch_transition_voltage=1e9,
        switch_transition_current=1e9,
    )

    dt = times[1] - times[0]
    sim_currents = []
    sim_voltages = []
    state = SimulationState()
    for _ in range(1, len(times)):
        circuit._step_rk4(state, dt)
        sim_currents.append(circuit.get_current())
        sim_voltages.append(circuit.get_voltage())

    assert np.allclose(
        np.abs(sim_currents), [abs(c) for c in currents[1:]], rtol=3e-1, atol=1e-2
    )
    assert np.allclose(sim_voltages, voltages[1:], rtol=3e-1, atol=1e-2)
