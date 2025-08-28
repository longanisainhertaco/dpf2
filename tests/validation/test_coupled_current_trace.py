import json
from pathlib import Path

from dpf2.core.bases import CouplingState
from dpf2.core.circuit import RLCCircuitSolver
from dpf2.physics.simple_plasma import ZeroDPlasma
from dpf2.diagnostics.synthetic_signals import coupled_current_waveform


def test_coupled_current_matches_reference():
    ref_path = Path(__file__).resolve().parents[2] / "ReferenceMaterial/coupled_current.json"
    ref_waveform = json.loads(ref_path.read_text())["waveform"]

    L_ext = 1e-6
    R_ext = 0.1
    C_ext = 1e-6
    V0_source = 0.0
    initial_voltage = 1000.0

    def inductance_model(t, current, voltage):
        t_end = 1e-6
        return (1e-7 * (1 + 50 * (t / t_end)), 0.0)

    plasma = ZeroDPlasma(inductance_model)
    circuit = RLCCircuitSolver(L_ext=L_ext, R_ext=R_ext, C_ext=C_ext, V0=V0_source)
    circuit.voltages[0] = initial_voltage

    current = circuit.currents[-1]
    voltage = circuit.voltages[-1]
    plasma.step(None, 0.0, current, voltage)

    dt = 1e-8
    states = []
    for _ in range(len(ref_waveform)):
        fb = plasma.coupling_interface()
        state = circuit.step(CouplingState(Lp=fb.Lp, emf=fb.emf, current=current, voltage=voltage), 0.0, dt)
        states.append(state)
        current, voltage = state.current, state.voltage
        plasma.step(None, dt, current, voltage)

    waveform = coupled_current_waveform(states)
    l1 = sum(abs(a - b) for a, b in zip(waveform, ref_waveform)) / len(ref_waveform)
    assert l1 < 1e-6
