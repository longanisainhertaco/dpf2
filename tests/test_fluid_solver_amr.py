import sys
import types
import numpy as np

# Stubs for heavy optional dependencies
amrex_stub = types.ModuleType("amrex")
amrex_stub.FabArrayIO = types.SimpleNamespace(tagCells=lambda *a, **k: 'tags')
amrex_stub.EBIndexSpace = types.SimpleNamespace(instance=lambda: types.SimpleNamespace(build_from_stl=lambda *_: None))
amrex_stub.MultiFab = object
amrex_stub.MultiFabLaplacian = object
amrex_stub.MLMG = object
sys.modules["amrex"] = amrex_stub

sys.modules.setdefault("adios2", types.ModuleType("adios2"))

numba_stub = types.ModuleType("numba")
numba_stub.njit = lambda *a, **k: (lambda f: f)
numba_stub.prange = range
numba_stub.cuda = types.SimpleNamespace()
sys.modules.setdefault("numba", numba_stub)
sys.modules.setdefault("numba.cuda", numba_stub.cuda)

sys.modules.setdefault("h5py", types.ModuleType("h5py"))

scipy_stub = types.ModuleType("scipy")
scipy_stub.__path__ = []
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
    def get_B(self):
        return np.zeros((4,4,4,3))

class DummyGeom:
    def __init__(self):
        self.refined_with = None
    def refine(self, tags):
        self.refined_with = tags

class DummyEOS:
    mean_ion_mass = 1.0
    gamma = 1.4
    def ion_pressure(self, rho, E):
        return E
    def electron_pressure(self, rho, E):
        return E


def test_amr_refine_invokes_geometry():
    solver = FluidSolverHighOrder.__new__(FluidSolverHighOrder)
    solver.state = {
        'density': DummyMF(np.ones((4,4,4))),
        'momentum': DummyMF(np.zeros((4,4,4,3))),
        'energy_i': DummyMF(np.ones((4,4,4))),
        'energy_e': DummyMF(np.ones((4,4,4)))
    }
    solver.field_manager = DummyFieldManager()
    solver.eos = DummyEOS()
    solver.dx = solver.dy = solver.dz = 1.0
    solver.do_amr = True
    solver.rho_threshold = 0.5
    solver.J_threshold = 0.5
    geom = DummyGeom()
    solver.geom = geom
    solver._interpolate_data = lambda: None

    np.any = lambda a: False
    np.isnan = lambda a: False
    np.linalg = types.SimpleNamespace(norm=lambda arr, axis=None: 0.0)
    np.gradient = lambda *args, **kwargs: np.zeros_like(args[0])

    import amrex
    amrex.FabArrayIO.tagCells = lambda *a, **k: 'tags'

    solver._amr_refine()
