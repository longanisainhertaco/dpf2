import numpy as np

from dpf2.physics import HallMHD
from dpf2.core.circuit import RLCCircuitSolver


def _shock_setup():
    model = HallMHD(hall_coeff=0.1)
    left = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    right = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    U_left = model.conservative_variables(left)
    U_right = model.conservative_variables(right)
    return model, U_left, U_right


def test_shock_propagation():
    model, U_left, U_right = _shock_setup()
    U = np.vstack([U_left, U_right])
    dx = 1.0
    dt = 0.1 * dx / max(model.max_speed(U_left, "x"), model.max_speed(U_right, "x"))
    for _ in range(5):
        U = model.ctu_update(U, dx, dt)

    assert U[1, 0] > U_right[0]
    for state in U:
        prim = model.primitive_variables(state)
        cons = model.conservative_variables(prim)
        assert np.allclose(cons, state)


def test_alfven_wave_propagation():
    model = HallMHD()
    n = 32
    x = np.arange(n)
    rho0 = 1.0
    B0 = 1.0
    amp = 1.0e-2
    By = amp * np.sin(2 * np.pi * x / n)
    vy = -amp * np.sin(2 * np.pi * x / n)

    U = np.zeros((n, 9))
    for i in range(n):
        prim = np.array([rho0, 0.0, vy[i], 0.0, 1.0, 0.0, By[i], B0])
        U[i] = model.conservative_variables(prim)

    dx = 1.0
    ca = B0 / np.sqrt(rho0)
    dt = 0.4 * dx / ca

    for _ in range(5):
        U = model.ctu_update(U, dx, dt, periodic=True)

    final_By = U[:, 6]
    assert np.isclose(np.max(np.abs(final_By)), amp, rtol=0.2)


def test_circuit_exchange():
    model, state, _ = _shock_setup()
    circuit = RLCCircuitSolver(L_ext=1.0, R_ext=0.0, C_ext=1.0, V0=0.5)
    circuit.voltages[-1] = 0.0
    dt = 1.0e-6
    current = 1.0

    state = model.step(state, dt, current=current, circuit=circuit)
    assert model.circuit_feedback is not None

    # coupling updated the circuit current
    assert model.current != current
