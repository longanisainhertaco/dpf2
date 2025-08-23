import numpy as np

from dpf2.hall_mhd_solver import HallMHDSolver, MHDState, _divergence


def _uniform_state(shape):
    rho = np.ones(shape)
    v = np.array([0.05, -0.1, 0.02])
    mom = rho[..., None] * v
    B = np.array([0.1, 0.05, -0.02]) * np.ones(shape + (3,))
    p = 1.0
    gamma = 5.0 / 3.0
    kinetic = 0.5 * np.sum(v**2)
    magnetic = 0.5 * np.sum(B[0, 0, 0] ** 2)
    energy = p / (gamma - 1.0) + kinetic + magnetic
    energy = np.full(shape, energy)
    return MHDState(rho=rho, mom=mom, energy=energy, B=B)


def test_energy_conservation():
    shape = (4, 4, 4)
    state = _uniform_state(shape)
    solver = HallMHDSolver()
    e0 = np.sum(state.energy)
    for _ in range(5):
        state = solver.step(state, 0.05)
    e1 = np.sum(state.energy)
    assert np.isclose(e0, e1)


def test_divergence_free_evolution():
    shape = (4, 4, 4)
    state = _uniform_state(shape)
    solver = HallMHDSolver()
    for _ in range(3):
        state = solver.step(state, 0.05)
        divB = _divergence(state.B)
        assert np.max(np.abs(divB)) < 1e-12

