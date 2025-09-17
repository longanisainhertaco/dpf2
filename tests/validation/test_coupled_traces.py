import json
from pathlib import Path
import numpy as np

from dpf2.core.bases import CouplingState
from dpf2.core.circuit import RLCCircuitSolver
from dpf2.physics.simple_plasma import ZeroDPlasma


def test_coupled_traces_match_reference():
    ref_path = (
        Path(__file__).resolve().parents[2] / "ReferenceMaterial/coupled_traces.json"
    )
    reference = json.loads(ref_path.read_text())

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
    num = len(reference["time"])
    states = []
    for _ in range(num):
        fb = plasma.coupling_interface()
        state = circuit.step(
            CouplingState(Lp=fb.Lp, emf=fb.emf, current=current, voltage=voltage),
            0.0,
            dt,
        )
        states.append(state)
        current, voltage = state.current, state.voltage
        plasma.step(None, dt, current, voltage)

    sim_time = [dt * (i + 1) for i in range(num)]
    sim_current = [s.current for s in states]
    sim_voltage = [s.voltage for s in states]

    atol = 0.0
    rtol = 1e-9

    assert np.allclose(sim_time, reference["time"], rtol=rtol, atol=atol)
    assert np.allclose(sim_current, reference["current"], rtol=rtol, atol=atol)
    assert np.allclose(sim_voltage, reference["voltage"], rtol=rtol, atol=atol)
