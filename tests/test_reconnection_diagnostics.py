import sys
import types
import numpy as np

# Provide minimal stubs for heavy optional dependencies
amrex_stub = types.ModuleType("amrex")
amrex_stub.EBIndexSpace = types.SimpleNamespace(instance=lambda: types.SimpleNamespace(build_from_stl=lambda *_: None))
amrex_stub.MultiFab = object
amrex_stub.MultiFabLaplacian = object
amrex_stub.MLMG = object
sys.modules.setdefault("amrex", amrex_stub)

sys.modules.setdefault("adios2", types.ModuleType("adios2"))

numba_stub = types.ModuleType("numba")
numba_stub.njit = lambda *a, **k: (lambda f: f)
numba_stub.prange = range
sys.modules.setdefault("numba", numba_stub)

collision_stub = types.ModuleType("dpf2.simulation.collision_model")
collision_stub.braginskii_coeffs = lambda *a, **k: (0, 0, 0, 0)
collision_stub.CollisionModel = object
sys.modules.setdefault("dpf2.simulation.collision_model", collision_stub)

sys.modules.setdefault("h5py", types.ModuleType("h5py"))

rad_stub = types.ModuleType("dpf2.simulation.radiation_model")
rad_stub.RadiationModel = lambda *a, **k: types.SimpleNamespace(compute_radiation_loss=lambda *a, **k: 0.0)
sys.modules.setdefault("dpf2.simulation.radiation_model", rad_stub)

from dpf2.simulation.fluid_solver_high_order import FluidSolverHighOrder

# Provide minimal linalg for numpy stub if needed
if not hasattr(np, "linalg"):
    import math

    def _norm(a, axis=None):
        data = np.array(a).data
        if axis is None:
            total = 0.0
            for plane in data:
                for row in plane:
                    for vec in row:
                        total += sum(v * v for v in vec)
            return math.sqrt(total)
        if axis == 3:
            out = []
            for plane in data:
                plane_out = []
                for row in plane:
                    plane_out.append([math.sqrt(sum(v * v for v in vec)) for vec in row])
                out.append(plane_out)
            return np.array(out)
        raise NotImplementedError

    np.linalg = types.SimpleNamespace(norm=_norm)


class DummyFieldManager:
    def __init__(self, E, B):
        self._E = E
        self._B = B
        self._J = np.zeros_like(E)

    def get_E(self):
        return self._E

    def get_B(self):
        return self._B

    def get_J(self):
        return self._J

    # Placeholder methods for compatibility
    def update_E(self, E):
        self._E = E

    def update_B(self, B):
        self._B = B

    def deposit_current(self, J):
        self._J = J


def test_reconnection_rate_zero():
    solver = FluidSolverHighOrder.__new__(FluidSolverHighOrder)
    fm = DummyFieldManager(np.zeros((1, 1, 1, 3)), np.zeros((1, 1, 1, 3)))
    solver.field_manager = fm
    rate = solver._reconnection_rate()
    if isinstance(rate, list):
        rate = rate[0][0]
    assert rate == 0.0


def test_reconnection_rate_parallel_fields():
    solver = FluidSolverHighOrder.__new__(FluidSolverHighOrder)
    E = np.zeros((1, 1, 1, 3))
    B = np.zeros((1, 1, 1, 3))
    E[0, 0, 0] = [1.0, 0.0, 0.0]
    B[0, 0, 0] = [2.0, 0.0, 0.0]
    fm = DummyFieldManager(E, B)
    solver.field_manager = fm
    rate = solver._reconnection_rate()
    if isinstance(rate, list):
        rate = rate[0][0]
    assert abs(rate - 2.0) < 1e-12
