import numpy as np
import pytest

from dpf2.physics import HallMHD
from dpf2.core.circuit import RLCCircuitSolver
from dpf2.mesh import Mesh3D
from dpf2.hall_mhd_solver import HallMHDSolver, MHDState
from dpf2.physics.hall_mhd import nrl_braginskii
from dpf2.diagnostics.quality_dashboard import QualityDashboard


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
    mesh = Mesh3D(0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1)
    dt = 0.1 * mesh.dx / max(
        model.max_speed(U_left, "x"), model.max_speed(U_right, "x")
    )
    for _ in range(5):
        U = model.ctu_update(U, mesh, dt)

    assert U[1, 0] > U_right[0]
    for state in U:
        prim = model.primitive_variables(state)
        cons = model.conservative_variables(prim)
        assert np.allclose(cons, state)


def test_alfven_wave_propagation():
    model = HallMHD()
    n = 32
    mesh = Mesh3D(0.0, float(n), 0.0, 1.0, 0.0, 1.0, n, 1, 1)
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

    ca = B0 / np.sqrt(rho0)
    dt = 0.4 * mesh.dx / ca

    for _ in range(5):
        U = model.ctu_update(U, mesh, dt, periodic=True)

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


def test_activation_gates_and_closure():
    state = MHDState(
        rho=np.ones((2, 1)),
        mom=np.zeros((2, 1, 3)),
        energy=np.ones((2, 1)) * 20.0,
        B=np.zeros((2, 1, 3)) + np.array([5.0, 0.0, 0.0]),
        eta=np.ones((2, 1)) * 1e-6,
    )
    solver = HallMHDSolver(hall_threshold=0.1, ei_threshold=0.2, scale_length=1e12)
    solver.step(state, 0.0)
    assert solver.hall_active
    assert solver.last_wce_tau_e > 0.1
    assert not solver.electron_inertia_active

    def closure(rho, T, B):
        return 1.0, 2.0

    low_rho_state = MHDState(
        rho=np.ones((2, 1)) * 1e-6,
        mom=np.zeros((2, 1, 3)),
        energy=np.ones((2, 1)),
        B=np.zeros((2, 1, 3)),
        eta=np.ones((2, 1)) * 1e-3,
    )
    solver2 = HallMHDSolver(braginskii=closure, scale_length=1e-3, ei_threshold=0.01)
    solver2.step(low_rho_state, 0.0)
    assert solver2.electron_inertia_active
    assert solver2.nu_par == 1.0
    assert solver2.kappa_par == 2.0

    nu, kappa = nrl_braginskii(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    assert nu.shape == (1,)
    assert kappa.shape == (1,)


def test_quality_diagnostics(tmp_path):
    B = np.zeros((2, 1, 3)) + np.array([5.0, 0.0, 0.0])
    state = MHDState(
        rho=np.ones((2, 1)) * 1e-6,
        mom=np.zeros((2, 1, 3)),
        energy=np.ones((2, 1)) * 20.0,
        B=B,
        eta=np.ones((2, 1)) * 1e-3,
    )
    q = QualityDashboard(output_dir=tmp_path)
    solver = HallMHDSolver(
        hall_threshold=0.1,
        ei_threshold=0.01,
        scale_length=1e-3,
        quality=q,
    )
    solver.step(state, 0.0)
    entry = q.history[-1]
    assert entry["hall_active"] is True
    assert entry["electron_inertia_active"] is True
    assert entry["hall_threshold"] == pytest.approx(0.1)
    assert entry["ei_threshold"] == pytest.approx(0.01)
    assert entry["wce_tau_e"] == pytest.approx(solver.last_wce_tau_e)
    assert entry["di_over_L"] == pytest.approx(solver.last_di_over_L)
