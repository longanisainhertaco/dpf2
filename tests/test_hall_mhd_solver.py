import numpy as np

from dpf2.hall_mhd_solver import HallMHDSolver, MHDState, _divergence


def _uniform_state(shape):
    rho = np.ones(shape)
    v = np.array([0.1, -0.2, 0.05])
    mom = rho[..., None] * v
    B = np.array([0.3, -0.1, 0.2]) * np.ones(shape + (3,))
    p = 1.0
    gamma = 5.0 / 3.0
    kinetic = 0.5 * np.sum(v**2)
    magnetic = 0.5 * np.sum(B[0, 0, 0] ** 2)
    energy = p / (gamma - 1.0) + kinetic + magnetic
    energy = np.full(shape, energy)
    return MHDState(rho=rho, mom=mom, energy=energy, B=B)


def test_conservation_and_divergence():
    shape = (4, 4, 4)
    state = _uniform_state(shape)
    solver = HallMHDSolver()
    new_state = solver.step(state, 0.1)
    assert np.allclose(new_state.rho, state.rho)
    assert np.allclose(new_state.mom, state.mom)
    assert np.allclose(new_state.energy, state.energy)
    assert np.allclose(new_state.B, state.B)
    assert np.max(np.abs(_divergence(new_state.B))) < 1e-12


def test_divergence_cleaning():
    shape = (4, 4, 4)
    rng = np.random.default_rng(0)
    rho = np.ones(shape)
    mom = np.zeros(shape + (3,))
    B = rng.random(shape + (3,)) - 0.5
    energy = np.ones(shape)
    state = MHDState(rho=rho, mom=mom, energy=energy, B=B)
    solver = HallMHDSolver()
    new_state = solver.step(state, 0.0)
    initial_div = np.max(np.abs(_divergence(B)))
    final_div = np.max(np.abs(_divergence(new_state.B)))
    assert initial_div > 1e-6
    assert final_div < 1e-12
