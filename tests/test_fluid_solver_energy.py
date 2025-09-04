import sys
import types
import numpy as np
import pytest

# Provide stubs for optional heavy dependencies
amrex_stub = types.ModuleType("amrex")
amrex_stub.EBIndexSpace = types.SimpleNamespace(instance=lambda: types.SimpleNamespace(build_from_stl=lambda *_: None))
amrex_stub.MultiFab = object
amrex_stub.MultiFabLaplacian = object
amrex_stub.MLMG = object
amrex_stub.FabArrayIO = types.SimpleNamespace(tagCells=lambda *a, **k: None)
sys.modules.setdefault("amrex", amrex_stub)

sys.modules.setdefault("adios2", types.ModuleType("adios2"))

numba_stub = types.ModuleType("numba")
numba_stub.njit = lambda *a, **k: (lambda f: f)
numba_stub.prange = range
numba_stub.cuda = types.SimpleNamespace()
sys.modules.setdefault("numba", numba_stub)
sys.modules.setdefault("numba.cuda", numba_stub.cuda)

sys.modules.setdefault("h5py", types.ModuleType("h5py"))

scipy_stub = types.ModuleType("scipy")
scipy_stub.__path__ = []  # mark as package
scipy_stub.constants = types.SimpleNamespace()
scipy_stub.interpolate = types.SimpleNamespace(
    interp1d=lambda *a, **k: None,
    RegularGridInterpolator=lambda *a, **k: None,
)
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.constants", scipy_stub.constants)
sys.modules.setdefault("scipy.interpolate", scipy_stub.interpolate)

from dpf2.simulation.fluid_solver_high_order import FluidSolverHighOrder


class DummyMF:
    def __init__(self, arr):
        self._arr = np.array(arr)
    def array(self):
        return self._arr
    def setVal(self, val):
        self._arr[:] = val

class DummyFieldManager:
    def __init__(self, B):
        self._B = B
    def get_B(self):
        return self._B


def make_solver():
    solver = FluidSolverHighOrder.__new__(FluidSolverHighOrder)
    solver.state = {
        'density': DummyMF(np.ones((2,2,2))),
        'momentum': DummyMF(np.zeros((2,2,2,3))),
        'energy_i': DummyMF(np.ones((2,2,2))),
        'energy_e': DummyMF(np.ones((2,2,2)))
    }
    solver.field_manager = DummyFieldManager(np.zeros((2,2,2,3)))
    solver.config = {'energy_tol': 1e-6}
    return solver


def test_total_energy_increment():
    solver = make_solver()
    e0 = solver.get_total_energy()
    solver.increment_internal_energy(0.5)
    e1 = solver.get_total_energy()
    size = 1
    for s in solver.state['density'].array().shape:
        size *= s
    assert e1 == pytest.approx(e0 + 0.5 * size * 2)


def test_energy_conservation_check():
    solver = make_solver()
    solver._check_energy_conservation(1.0, 1.0 + 5e-7)  # within tolerance
    with pytest.raises(RuntimeError):
        solver._check_energy_conservation(1.0, 1.1)
