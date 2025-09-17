import random
import sys
import types

# Stub optional heavy dependencies used by radiation_model
sys.modules.setdefault("amrex", types.ModuleType("amrex"))
sys.modules.setdefault("adios2", types.ModuleType("adios2"))
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
scipy_stub = types.ModuleType("scipy")
interpolate_stub = types.ModuleType("interpolate")
interpolate_stub.RegularGridInterpolator = lambda *args, **kwargs: None
scipy_stub.interpolate = interpolate_stub
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.interpolate", interpolate_stub)
numba_stub = types.ModuleType("numba")
numba_stub.njit = lambda *args, **kwargs: (lambda f: f)
numba_stub.prange = range
sys.modules.setdefault("numba", numba_stub)

# Provide minimal numpy stub with required functions
import math
import numpy_stub

numpy_stub.np.sin = math.sin
numpy_stub.np.cos = math.cos
numpy_stub.np.arccos = math.acos
numpy_stub.np.linalg = types.SimpleNamespace(
    norm=lambda v: math.sqrt(sum(x * x for x in v))
)
sys.modules["numpy"] = numpy_stub.np

from dpf2.simulation.radiation_model import Photon, m_e, c

_norm = numpy_stub.np.linalg.norm


def test_photon_scatter_rotates_direction():
    random.seed(0)
    p = Photon(pos=[0, 0, 0], dir=[1, 0, 0], energy=m_e * c**2, group=0)
    initial_dir = p.dir.copy()
    p.scatter()
    changed = any(abs(a - b) > 1e-12 for a, b in zip(p.dir, initial_dir))
    assert changed
    norm = _norm(p.dir)
    assert abs(norm - 1.0) < 1e-12
