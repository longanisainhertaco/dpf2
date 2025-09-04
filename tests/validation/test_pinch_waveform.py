"""Regression test for synthetic pinch waveform."""

import numpy as np
from dpf2.core.circuit import RLCCircuitSolver, CouplingState


def test_pinch_waveform_current_dip_and_voltage_spike():
    solver = RLCCircuitSolver(L_ext=1e-6, R_ext=0.1, C_ext=1e-6, V0=5.0)
    solver.voltages[0] = 0.0
    state = CouplingState(Lp=0.0, emf=0.0, current=0.0, voltage=0.0)
    dt = 1e-8
    steps = 100
    pinch_step = 50
    Lp_spike = 1e-5
    for i in range(steps):
        Lp = Lp_spike if i == pinch_step else 0.0
        state = solver.step(
            CouplingState(Lp=Lp, emf=0.0, current=state.current, voltage=state.voltage),
            0.0,
            dt,
        )
    currents = np.array(solver.currents)
    voltages = np.array(solver.voltages)
    assert currents[pinch_step + 1] < currents[pinch_step]
    assert abs(voltages[pinch_step + 1]) > abs(voltages[pinch_step])
