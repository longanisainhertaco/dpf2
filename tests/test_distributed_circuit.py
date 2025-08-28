import sys
from pathlib import Path
import importlib.util
import types
import numpy as np
import pytest

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
    pass
class FieldManager:
    """Placeholder field manager."""
    pass
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
SwitchModel = circuit_mod.SwitchModel


class DummyCollision:
    def spitzer_resistivity(self, *args, **kwargs):
        return 0.0


class DummyFieldManager:
    def get_J(self):
        return 0.0


def _make_state():
    state = SimulationState()
    # Provide required attributes for plasma inductance/resistance
    state.sheath_position = 0.1  # m
    state.electron_temperature = 1e3  # K
    state.density = 1e20  # kg/m^3 (arbitrary positive)
    return state


def test_distributed_segments_match_lumped_and_energy_conserved():
    state = _make_state()
    params = dict(
        C=1e-6,
        L0=1e-6,
        R0=1.0,
        anode_radius=0.01,
        cathode_radius=0.02,
        collision_model=DummyCollision(),
        field_manager=DummyFieldManager(),
        V0=1000.0,
        ESR=0.1,
        ESL=1e-9,
        parasitic_inductance=1e-9,
        stray_capacitance=1e-9,
        switch_initial_resistance=1e-3,
        switch_final_resistance=1e-3,
        switch_transition_voltage=1e9,
        switch_transition_current=1e9,
    )

    lumped = CircuitModel(**params, transmission_line=False)

    distributed = CircuitModel(
        **params,
        transmission_line=True,
        transmission_line_impedance=50.0,
        transmission_line_length=1e-3,
        transmission_line_velocity_factor=1.0,
    )
    # reduce computational burden
    distributed.transmission_line_model.num_segments = 5

    dt = 1e-9
    steps = 50
    curr_l, volt_l, curr_d, volt_d = [], [], [], []

    for _ in range(steps):
        lumped._step_rk4(state, dt)
        distributed._step_rk4(state, dt)
        curr_l.append(lumped.get_current())
        volt_l.append(lumped.get_voltage())
        curr_d.append(distributed.get_current())
        volt_d.append(distributed.get_voltage())

    assert np.allclose(curr_l, curr_d, rtol=1e-2, atol=1e-4)
    assert np.allclose(volt_l, volt_d, rtol=1e-2, atol=1e-4)

    # Energy conservation for lumped circuit
    R_tot = lumped.R0 + lumped.ESR + lumped.switch_model.get_resistance(0.0, 0.0, 0.0)
    L_tot = lumped.L0 + lumped.ESL + lumped.Ls + lumped.plasma_inductance(state)
    C_main = lumped.C
    Cs = lumped.Cs

    initial_energy = 0.5 * (C_main + Cs) * (params["V0"] ** 2)
    dissipated = sum((I ** 2) * R_tot * dt for I in curr_l)
    final_I = curr_l[-1]
    final_V = volt_l[-1]
    stored = 0.5 * (C_main + Cs) * (final_V ** 2) + 0.5 * L_tot * (final_I ** 2)

    assert np.isclose(initial_energy, stored + dissipated, rtol=1e-2)


def test_switch_triggering_edge_cases():
    state = _make_state()
    # Case where switch should not trigger
    circuit_off = CircuitModel(
        C=1e-6,
        L0=1e-6,
        R0=1.0,
        anode_radius=0.01,
        cathode_radius=0.02,
        collision_model=DummyCollision(),
        field_manager=DummyFieldManager(),
        V0=10.0,
        switch_initial_resistance=1e6,
        switch_final_resistance=1.0,
        switch_transition_voltage=100.0,
        switch_transition_current=1e6,
    )
    circuit_off._step_rk4(state, 1e-9)
    assert not circuit_off.switch_model.is_closed

    # Case where switch should trigger immediately
    circuit_on = CircuitModel(
        C=1e-6,
        L0=1e-6,
        R0=1.0,
        anode_radius=0.01,
        cathode_radius=0.02,
        collision_model=DummyCollision(),
        field_manager=DummyFieldManager(),
        V0=1000.0,
        switch_initial_resistance=1e6,
        switch_final_resistance=1.0,
        switch_transition_voltage=100.0,
        switch_transition_current=1.0,
        switch_transition_time=0.0,
    )
    circuit_on._step_rk4(state, 1e-9)
    assert circuit_on.switch_model.is_closed


def test_parasitic_elements_affect_dynamics():
    state = _make_state()
    base = CircuitModel(
        C=1e-6,
        L0=1e-6,
        R0=1.0,
        anode_radius=0.01,
        cathode_radius=0.02,
        collision_model=DummyCollision(),
        field_manager=DummyFieldManager(),
        V0=1000.0,
        switch_initial_resistance=1e-3,
        switch_final_resistance=1e-3,
        switch_transition_voltage=1e9,
        switch_transition_current=1e9,
    )

    with_parasitics = CircuitModel(
        C=1e-6,
        L0=1e-6,
        R0=1.0,
        anode_radius=0.01,
        cathode_radius=0.02,
        collision_model=DummyCollision(),
        field_manager=DummyFieldManager(),
        V0=1000.0,
        parasitic_inductance=1e-3,
        stray_capacitance=1e-3,
        switch_initial_resistance=1e-3,
        switch_final_resistance=1e-3,
        switch_transition_voltage=1e9,
        switch_transition_current=1e9,
    )

    dt = 1e-9
    for _ in range(200):
        base._step_rk4(state, dt)
        with_parasitics._step_rk4(state, dt)

    assert not np.isclose(base.get_current(), with_parasitics.get_current())
    assert not np.isclose(base.get_voltage(), with_parasitics.get_voltage())
