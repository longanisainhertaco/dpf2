import numpy as np

from dpf2.hall_mhd_solver import HallMHDSolver, MHDState


def _alfven_wave_state(nx: int) -> MHDState:
    x = np.arange(nx)
    rho = np.ones((nx, 1, 1))
    B0 = 1.0
    B = np.zeros((nx, 1, 1, 3))
    B[..., 0] = B0
    perturb = 1e-3 * np.sin(2 * np.pi * x / nx)[:, None, None]
    B[..., 1] = perturb
    v = np.zeros_like(B)
    v[..., 1] = perturb / np.sqrt(rho)
    mom = rho[..., None] * v
    gamma = 5.0 / 3.0
    p = np.ones((nx, 1, 1))
    kinetic = 0.5 * rho * np.sum(v**2, axis=-1)
    magnetic = 0.5 * np.sum(B**2, axis=-1)
    energy = p / (gamma - 1.0) + kinetic + magnetic
    return MHDState(rho=rho, mom=mom, energy=energy, B=B)


def test_alfven_wave_propagation():
    nx = 32
    state = _alfven_wave_state(nx)
    solver = HallMHDSolver()
    dt = 0.1
    new_state = solver.step(state, dt)

    vA = 1.0 / np.sqrt(1.0)
    shift = int(round(vA * dt))
    analytic = np.roll(state.B[..., 1], -shift, axis=0)
    assert np.allclose(new_state.B[..., 1], analytic, atol=1e-4)
