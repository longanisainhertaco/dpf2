import numpy as np

from dpf2.hall_mhd_solver import HallMHDSolver, MHDState


def _state(shape):
    rho = np.ones(shape)
    v = np.zeros(3)
    mom = rho[..., None] * v
    B = np.zeros(shape + (3,))
    energy = np.ones(shape)
    return MHDState(rho=rho, mom=mom, energy=energy, B=B)


def test_bc_and_amr_hooks_invoked():
    shape = (2, 2, 2)
    state = _state(shape)
    calls = {"bc": 0, "amr": 0}

    def bc(s):
        calls["bc"] += 1
        s.rho[0, 0, 0] = 2.0

    def refine(s):
        calls["amr"] += 1

    solver = HallMHDSolver(bc=bc, refine=refine)
    new_state = solver.step(state, 0.01)

    assert calls["bc"] == 2  # called before and after step
    assert calls["amr"] == 1
    assert new_state.rho[0, 0, 0] == 2.0
