import numpy as np

from dpf2.hall_mhd_solver import HallMHDSolver, MHDState, _divergence


def _state(shape):
    """Create a simple uniform state for tests."""
    rho = np.ones(shape)
    mom = np.zeros(shape + (3,))
    B = np.zeros(shape + (3,))
    energy = np.ones(shape)
    return MHDState(rho=rho, mom=mom, energy=energy, B=B)


def test_braginskii_transport_and_ct():
    shape = (4, 4, 4)
    x = np.arange(shape[0]).reshape(shape[0], 1, 1)
    # Base state with gradients along x and magnetic field in x-direction
    rho = np.ones(shape)
    mom = np.zeros(shape + (3,))
    mom[..., 0] = x[..., 0, 0]
    B = np.zeros(shape + (3,))
    B[..., 0] = 1.0
    energy = 1.0 + 0.1 * x[..., 0, 0]
    state = MHDState(rho=rho, mom=mom, energy=energy, B=B)

    base_solver = HallMHDSolver()
    br_solver = HallMHDSolver(braginskii=lambda r, T, Bmag: (0.5, 0.5))

    out_base = base_solver.step(state, 0.01)
    out_brag = br_solver.step(
        MHDState(rho=rho.copy(), mom=mom.copy(), energy=energy.copy(), B=B.copy()), 0.01
    )

    # Braginskii coefficients should modify momentum and energy
    assert not np.allclose(out_base.mom, out_brag.mom)
    assert not np.allclose(out_base.energy, out_brag.energy)
    # Constrained transport keeps the field divergence free
    assert np.max(np.abs(_divergence(out_brag.B))) < 1e-12


def test_mpi_decomposition_and_amr_hook(monkeypatch):
    from dpf2 import hall_mhd_solver as solver_mod

    class DummyMPI:
        PROC_NULL = -1

        @staticmethod
        def Compute_dims(size, dims):
            return (1, 1, 1)

    class DummyComm:
        def __init__(self):
            self.calls = 0

        def Get_size(self):
            return 1

        def Create_cart(self, dims, periods):
            return self

        def Shift(self, axis, disp):
            return (DummyMPI.PROC_NULL, DummyMPI.PROC_NULL)

        def Sendrecv(self, sendbuf, dest, recvbuf, source):
            self.calls += 1

    monkeypatch.setattr(solver_mod, "MPI", DummyMPI)
    comm = DummyComm()
    amr_calls = {"count": 0}

    def refine(state):
        amr_calls["count"] += 1

    solver = solver_mod.HallMHDSolver(comm=comm, refine=refine)
    state = _state((2, 2, 2))
    solver.step(state, 0.0)

    assert solver.cart_comm is comm
    assert comm.calls > 0  # ghost-cell exchange invoked
    assert amr_calls["count"] == 2  # before and after step
