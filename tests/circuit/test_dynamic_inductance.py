import numpy as np

from dpf2.physics.hall_mhd import HallMHD
from dpf2.circuit_solver import CircuitSolver, RLCCircuit
from dpf2.diagnostics.performance_metrics import reconstruct_plasma_inductance


def test_dynamic_inductance_reconstruction():
    # time grid
    dt = 1e-7
    t_end = 5e-6
    n_steps = int(t_end / dt)
    times = np.array([i * dt for i in range(n_steps)])

    # simple circuit parameters
    L_ext = 1e-6
    R = 0.1
    C = 1e-3
    V0 = 1000.0

    circuit = CircuitSolver(RLCCircuit(L=L_ext, R=R, C=C, V0=V0))
    # start with zero capacitor voltage so the supply drives the current
    circuit.voltages[0] = 0.0
    plasma = HallMHD()

    state = np.zeros(9)
    current = 1e-3
    voltage = 0.0

    def lp_func(t: float) -> float:
        return 1e-6 + 5e-7 * t / t_end

    currents = []
    voltages = []  # capacitor voltage history
    energies = []

    for t in times:
        Lp = lp_func(t)
        B = current * np.sqrt(Lp)
        state[5:8] = (B, 0.0, 0.0)

        currents.append(current)
        voltages.append(voltage)
        energies.append(0.5 * B * B)

        plasma.step(state, dt, current=current, voltage=voltage, circuit=circuit)
        current = plasma.circuit_feedback.current
        voltage = plasma.circuit_feedback.voltage

    # Effective circuit voltage is the drive minus capacitor drop
    v_eff = np.array([V0 - v for v in voltages])
    res = reconstruct_plasma_inductance(
        times,
        np.array(currents),
        v_eff,
        np.array(energies),
        resistance=R,
        external_inductance=L_ext,
    )

    expected = np.array([lp_func(t) for t in times])
    assert np.allclose(res["Lp_field"], expected, rtol=1e-6, atol=1e-9)
    assert np.allclose(res["Lp_circuit"], expected, rtol=5e-2)
