
import sys
from pathlib import Path
import importlib.util
import types
import numpy as np
import cmath
import math
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

# Minimal core stubs required by ``CircuitModel``
core_stub = types.ModuleType("dpf2.core")
bases_stub = types.ModuleType("dpf2.core.bases")

class CouplingState:
    """Lightweight replacement for the real ``CouplingState`` dataclass."""

    def __init__(self, Lp: float = 0.0, emf: float = 0.0, current: float = 0.0, voltage: float = 0.0):
        self.Lp = Lp
        self.emf = emf
        self.current = current
        self.voltage = voltage

bases_stub.CouplingState = CouplingState
core_stub.bases = bases_stub
sys.modules["dpf2.core"] = core_stub
sys.modules["dpf2.core.bases"] = bases_stub

module_path = Path(__file__).resolve().parent.parent / "src/dpf2/simulation/circuit.py"
spec = importlib.util.spec_from_file_location("dpf2.simulation.circuit", module_path)
circuit_mod = importlib.util.module_from_spec(spec)
sys.modules["dpf2.simulation.circuit"] = circuit_mod
spec.loader.exec_module(circuit_mod)
CircuitModel = circuit_mod.CircuitModel
SwitchModel = circuit_mod.SwitchModel

# Remove the temporary package stubs so that other tests can import the real package
sys.modules.pop("dpf2", None)
sys.modules.pop("dpf2.simulation", None)
sys.modules.pop("dpf2.core", None)
sys.modules.pop("dpf2.core.bases", None)
sys.modules.pop("dpf2.simulation.utils", None)
sys.modules.pop("dpf2.simulation.constants", None)


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


# ---------------------------------------------------------------------------
# New tests exercising the light‑weight distributed circuit solver

# Load the lightweight distributed circuit utilities directly from the source
module_path = Path(__file__).resolve().parent.parent / "src/dpf2/circuit/distributed.py"
spec = importlib.util.spec_from_file_location("dpf2.circuit.distributed", module_path)
dist_mod = importlib.util.module_from_spec(spec)
sys.modules["dpf2.circuit.distributed"] = dist_mod
spec.loader.exec_module(dist_mod)
TransmissionLineSegment = dist_mod.TransmissionLineSegment
TriggeredSwitch = dist_mod.TriggeredSwitch

module_path_rlc = Path(__file__).resolve().parent.parent / "src/dpf2/rlc_solver.py"
spec_rlc = importlib.util.spec_from_file_location("dpf2.rlc_solver", module_path_rlc)
rlc_mod = importlib.util.module_from_spec(spec_rlc)
sys.modules["dpf2.rlc_solver"] = rlc_mod
spec_rlc.loader.exec_module(rlc_mod)
solve_distributed_circuit = rlc_mod.solve_distributed_circuit


def test_lumped_vs_distributed_segments_and_energy():
    """Two half segments should behave like a single lumped element."""

    V0 = 1000.0
    L_tot = 1e-6
    C_tot = 1e-6

    lumped = [
        TransmissionLineSegment(
            from_node=0,
            to_node=1,
            length=1.0,
            L_per_m=L_tot,
            R_per_m=0.0,
            C_per_m=C_tot,
        )
    ]

    distributed = [
        TransmissionLineSegment(
            from_node=0,
            to_node=1,
            length=0.5,
            L_per_m=L_tot,
            R_per_m=0.0,
            C_per_m=C_tot,
        ),
        TransmissionLineSegment(
            from_node=1,
            to_node=2,
            length=0.5,
            L_per_m=L_tot,
            R_per_m=0.0,
            C_per_m=C_tot,
        ),
    ]

    res_l = solve_distributed_circuit(lumped, None, V0=V0, t_end=1e-6, dt=1e-8)
    res_d = solve_distributed_circuit(distributed, None, V0=V0, t_end=1e-6, dt=1e-8)

    assert np.allclose(res_l.current, res_d.current, rtol=1e-3, atol=1e-6)
    assert np.allclose(res_l.voltage, res_d.voltage, rtol=1e-3, atol=1e-6)

    # With zero resistance the system energy should remain constant
    initial = 0.5 * C_tot * V0**2
    final = 0.5 * C_tot * res_d.voltage[-1] ** 2 + 0.5 * L_tot * res_d.current[-1] ** 2
    assert np.isclose(initial, final, rtol=1e-2)


def test_switch_alias_import():
    """Compatibility layer should expose ``Switch`` as an alias."""

    module_path = Path(__file__).resolve().parent.parent / "src/dpf2/distributed_circuit.py"
    spec = importlib.util.spec_from_file_location("dpf2.distributed_circuit", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dpf2.distributed_circuit"] = mod
    spec.loader.exec_module(mod)

    assert mod.Switch is mod.TriggeredSwitch


def test_branched_network_current_split_and_energy():
    """Currents in symmetric branches should split equally and conserve energy."""

    V0 = 1000.0
    C_main = 1e-6
    L_series = 1e-6
    L_branch = 1e-6

    segments = [
        # Series inductor between the capacitor and the branching node
        TransmissionLineSegment(0, 1, length=1.0, L_per_m=L_series, R_per_m=0.0, C_per_m=0.0),
        # Two identical branches in parallel
        TransmissionLineSegment(1, 2, length=1.0, L_per_m=L_branch, R_per_m=0.0, C_per_m=0.0),
        TransmissionLineSegment(1, 2, length=1.0, L_per_m=L_branch, R_per_m=0.0, C_per_m=0.0),
        # Capacitor from source to ground providing the initial energy
        TransmissionLineSegment(0, 2, length=1.0, L_per_m=0.0, R_per_m=0.0, C_per_m=C_main),
    ]

    res = solve_distributed_circuit(segments, None, V0=V0, t_end=1e-7, dt=1e-9)

    # Branch currents for the two parallel paths should be identical
    branch1 = res.branch_currents[:, 1]
    branch2 = res.branch_currents[:, 2]
    assert np.allclose(branch1, branch2, rtol=1e-3, atol=1e-6)

    # Total energy in the lossless system should remain constant
    L_vals = np.array([segments[0].totals()[0], segments[1].totals()[0], segments[2].totals()[0]])
    initial = 0.5 * C_main * V0**2
    final = 0.5 * C_main * res.voltage[-1] ** 2 + 0.5 * sum(
        L_vals[i] * (res.branch_currents[-1, i] ** 2) for i in range(3)
    )
    assert np.isclose(initial, final, rtol=1e-3)


def test_transmission_line_phase_and_attenuation_across_frequencies():
    seg = TransmissionLineSegment(
        0,
        1,
        length=1.0,
        L_per_m=1e-6,
        R_per_m=1e-3,
        C_per_m=1e-9,
        skin_effect_coeff=1e-3,
        dielectric_loss_coeff=1e-4,
    )

    f1 = 1e6
    f2 = 5e6
    res1 = solve_distributed_circuit([seg], None, V0=1.0, t_end=5e-6, dt=1e-8, frequency=f1)
    res2 = solve_distributed_circuit([seg], None, V0=1.0, t_end=5e-6, dt=1e-8, frequency=f2)

    def _measure(res, freq):
        vals_out = [row[1] for row in res.node_voltages]
        vals_in = [row[0] for row in res.node_voltages]
        amp = (max(vals_out) - min(vals_out)) / 2.0
        period = int((1.0 / freq) / (res.t[1] - res.t[0]))
        vals_in_period = vals_in[:period]
        vals_out_period = vals_out[:period]
        idx_in = max(range(len(vals_in_period)), key=lambda i: vals_in_period[i])
        idx_out = max(range(len(vals_out_period)), key=lambda i: vals_out_period[i])
        delay = res.t[idx_out] - res.t[idx_in]
        phase = -2.0 * np.pi * freq * delay
        return amp, phase

    amp1, phase1 = _measure(res1, f1)
    amp2, phase2 = _measure(res2, f2)

    H1 = cmath.exp(-seg.propagation_constant(f1) * seg.length)
    H2 = cmath.exp(-seg.propagation_constant(f2) * seg.length)
    assert np.isclose(amp1, abs(H1), rtol=1e-2)
    assert np.isclose(amp2, abs(H2), rtol=1e-2)
    assert np.isclose(phase1, cmath.phase(H1), rtol=1e-2, atol=2e-2)
    assert np.isclose(phase2, cmath.phase(H2), rtol=1e-2, atol=1e-1)


def test_reflection_coefficient_frequency_dependence():
    seg = TransmissionLineSegment(
        0,
        1,
        length=1.0,
        L_per_m=1e-6,
        R_per_m=1e-3,
        C_per_m=1e-9,
        skin_effect_coeff=1e-3,
        dielectric_loss_coeff=1e-4,
    )
    ZL = 75.0
    f1, f2 = 1e6, 5e6
    r1 = seg.reflection_coefficient(f1, ZL)
    r2 = seg.reflection_coefficient(f2, ZL)

    def _manual(freq):
        w = 2.0 * np.pi * freq
        R = seg.R_per_m + seg.skin_effect_coeff * math.sqrt(freq)
        L = seg.L_per_m
        C = seg.C_per_m
        loss = seg.dielectric_loss_coeff * math.sqrt(freq)
        G = w * C * loss
        Z0 = cmath.sqrt((R + 1j * w * L) / (G + 1j * w * C))
        return (ZL - Z0) / (ZL + Z0)

    assert abs(r1 - _manual(f1)) < 1e-12
    assert abs(r2 - _manual(f2)) < 1e-12
    assert not np.isclose(abs(r1), abs(r2))
def _measure_amp_phase(node_voltages, t, freq):
    vals_out = [row[1] for row in node_voltages]
    vals_in = [row[0] for row in node_voltages]
    amp = (max(vals_out) - min(vals_out)) / 2.0
    period = int((1.0 / freq) / (t[1] - t[0]))
    vals_in_period = vals_in[:period]
    vals_out_period = vals_out[:period]
    idx_in = max(range(len(vals_in_period)), key=lambda i: vals_in_period[i])
    idx_out = max(range(len(vals_out_period)), key=lambda i: vals_out_period[i])
    delay = t[idx_out] - t[idx_in]
    phase = -2.0 * np.pi * freq * delay
    return amp, phase


def test_single_segment_reflection_and_phase():
    seg = TransmissionLineSegment(
        from_node=0,
        to_node=1,
        length=1.0,
        L_per_m=2.5e-7,
        R_per_m=0.0,
        C_per_m=1e-10,
    )
    freq = 1e7
    ZL = 25.0
    sol = solve_distributed_circuit([seg], [], V0=1.0, t_end=2e-6, dt=1e-9, frequency=freq, Z_load=ZL)

    gamma = seg.propagation_constant(freq)
    Z0 = seg.characteristic_impedance(freq)
    H = 1.0 / (cmath.cosh(gamma * seg.length) + (Z0 / ZL) * cmath.sinh(gamma * seg.length))
    amp_exp = abs(H)
    phase_exp = cmath.phase(H)

    amp_sol, phase_sol = _measure_amp_phase(sol.node_voltages, sol.t, freq)
    assert np.isclose(amp_sol, amp_exp, rtol=1e-3, atol=1e-6)
    assert np.isclose(phase_sol, phase_exp, rtol=1e-1, atol=1e-6)
    refl = (ZL - Z0) / (ZL + Z0)
    assert np.isclose(sol.reflections[0], refl, rtol=1e-6, atol=1e-6)


def test_two_segment_interface_reflection():
    seg1 = TransmissionLineSegment(
        from_node=0,
        to_node=1,
        length=0.5,
        L_per_m=2.5e-7,
        R_per_m=0.0,
        C_per_m=1e-10,
    )
    seg2 = TransmissionLineSegment(
        from_node=1,
        to_node=2,
        length=0.5,
        L_per_m=5.625e-7,
        R_per_m=0.0,
        C_per_m=1e-10,
    )
    freq = 1e7
    ZL = seg2.characteristic_impedance(freq)
    sol = solve_distributed_circuit([seg1, seg2], [], V0=1.0, t_end=2e-6, dt=1e-9, frequency=freq, Z_load=ZL)

    Z0_1 = seg1.characteristic_impedance(freq)
    Z0_2 = seg2.characteristic_impedance(freq)
    gamma1 = seg1.propagation_constant(freq) * seg1.length
    gamma2 = seg2.propagation_constant(freq) * seg2.length
    M1 = [
        [cmath.cosh(gamma1), Z0_1 * cmath.sinh(gamma1)],
        [cmath.sinh(gamma1) / Z0_1, cmath.cosh(gamma1)],
    ]
    M2 = [
        [cmath.cosh(gamma2), Z0_2 * cmath.sinh(gamma2)],
        [cmath.sinh(gamma2) / Z0_2, cmath.cosh(gamma2)],
    ]
    # Manual 2x2 matrix multiplication
    A = M1[0][0] * M2[0][0] + M1[0][1] * M2[1][0]
    B = M1[0][0] * M2[0][1] + M1[0][1] * M2[1][1]
    C = M1[1][0] * M2[0][0] + M1[1][1] * M2[1][0]
    D = M1[1][0] * M2[0][1] + M1[1][1] * M2[1][1]
    H = 1.0 / (A + B / ZL)
    amp_exp = abs(H)
    phase_exp = cmath.phase(H)
    amp_sol, phase_sol = _measure_amp_phase(sol.node_voltages, sol.t, freq)
    assert np.isclose(amp_sol, amp_exp, rtol=1e-3, atol=1e-6)
    assert np.isclose(phase_sol, phase_exp, rtol=1e-1, atol=1e-6)
    refl_int = (Z0_2 - Z0_1) / (Z0_2 + Z0_1)
    assert np.isclose(sol.reflections[0], refl_int, rtol=1e-6, atol=1e-6)
    assert np.isclose(sol.reflections[1], 0.0, atol=1e-12)

