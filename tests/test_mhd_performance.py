import time
import numpy as np

from dpf2.hall_mhd_solver import HallMHDSolver, MHDState


def _make_state(n: int) -> MHDState:
    shape = (n, n, n)
    rho = np.full(shape, 1.0)
    mom = np.zeros(shape + (3,))
    B = np.zeros(shape + (3,))
    energy = np.full(shape, 1.0)
    return MHDState(rho=rho, mom=mom, energy=energy, B=B)


def _run_solver(n: int, steps: int = 5) -> float:
    solver = HallMHDSolver()
    state = _make_state(n)
    start = time.perf_counter()
    for _ in range(steps):
        state = solver.step(state, 1e-6)
    return time.perf_counter() - start


def test_mhd_solver_scaling():
    sizes = [4, 8, 16]
    times = [_run_solver(n) for n in sizes]
    # ensure larger grids take longer to compute
    assert times[0] < times[-1]
    assert times[1] < times[-1]
