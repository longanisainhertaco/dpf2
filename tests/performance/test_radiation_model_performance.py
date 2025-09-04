import sys
import types
import time
import math

# Minimal stubs for optional dependencies
sys.modules.setdefault("amrex", types.ModuleType("amrex"))
sys.modules.setdefault("adios2", types.ModuleType("adios2"))
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
scipy_stub = types.ModuleType("scipy")
interpolate_stub = types.ModuleType("interpolate")
interpolate_stub.RegularGridInterpolator = lambda *a, **k: None
scipy_stub.interpolate = interpolate_stub
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.interpolate", interpolate_stub)
numba_stub = types.ModuleType("numba")
numba_stub.njit = lambda *a, **k: (lambda f: f)
numba_stub.prange = range
sys.modules.setdefault("numba", numba_stub)

import numpy_stub

numpy_stub.np.any = lambda arr: any(arr)
numpy_stub.np.linalg = types.SimpleNamespace(norm=lambda v: numpy_stub.sqrt(sum(x * x for x in v)))
numpy_stub.np.arccos = math.acos
numpy_stub.np.cos = math.cos
numpy_stub.np.sin = math.sin
numpy_stub.np.log = math.log
sys.modules["numpy"] = numpy_stub.np
np = numpy_stub.np

from dpf2.simulation.radiation_model import RadiationModel, Photon, m_e, c


def test_level_population_performance():
    rm = RadiationModel.__new__(RadiationModel)
    rm.level_pop = None
    Te = np.ones((5, 5, 5))
    ne = np.ones((5, 5, 5))
    start = time.perf_counter()
    for _ in range(100):
        rm._update_level_population(Te, ne, dt=1e-9)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


def test_pair_production_performance():
    rm = RadiationModel.__new__(RadiationModel)
    rm.pairs_created = 0
    photons = [Photon([0, 0, 0], [1, 0, 0], 3 * m_e * c ** 2, 0) for _ in range(100)]
    start = time.perf_counter()
    for p in photons:
        rm._pair_production(p, dt=1e-9)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0

