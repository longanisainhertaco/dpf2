import random
import sys
import types
import math

# Stub optional heavy dependencies used by radiation_model
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

# provide small helpers missing from numpy_stub
numpy_stub.np.any = lambda arr: any(arr)
numpy_stub.np.linalg = types.SimpleNamespace(
    norm=lambda v: numpy_stub.sqrt(sum(x * x for x in v))
)
numpy_stub.np.arccos = math.acos
numpy_stub.np.cos = math.cos
numpy_stub.np.sin = math.sin
numpy_stub.np.log = math.log
sys.modules["numpy"] = numpy_stub.np
np = numpy_stub.np

from dpf2.simulation.radiation_model import Photon, RadiationModel, m_e, c


def test_polarization_rotates_and_is_orthogonal():
    random.seed(0)
    p = Photon([0, 0, 0], [1, 0, 0], m_e * c**2, 0, polarization=[0, 1, 0])
    p.scatter()
    assert abs(np.dot(p.dir, p.polarization)) < 1e-12


def test_pair_production_counts_pairs():
    rm = RadiationModel.__new__(RadiationModel)
    rm.pairs_created = 0
    photon = Photon([0, 0, 0], [1, 0, 0], 3 * m_e * c**2, 0)
    orig = random.random
    random.random = lambda: 0.0
    occurred = rm._pair_production(photon, dt=1e30)
    random.random = orig
    assert occurred
    assert rm.pairs_created == 1


def test_non_lte_population_time_dependence():
    rm = RadiationModel.__new__(RadiationModel)
    rm.level_pop = None
    rm.non_lte_line_transport = True
    Te = np.ones((1, 1, 1))
    ne = np.ones((1, 1, 1))
    pop1 = rm._update_level_population(Te, ne, dt=1.0)
    pop2 = rm._update_level_population(Te, ne, dt=1.0)
    assert pop1[0][0][0] != pop2[0][0][0]
