import sys
import types
import numpy as np
import pytest


# Skip tests if optional radiation dependencies are missing
pytest.importorskip("amrex")
pytest.importorskip("adios2")

from dpf2.simulation.radiation_model import RadiationModel

import h5py_stub as h5py

# Provide a minimal numpy stub for the methods used in these tests
np_stub = types.SimpleNamespace(
    array=lambda x, dtype=None: x,
    power=lambda a, b: a**b,
    allclose=lambda a, b, rtol=1e-5, atol=1e-8: abs(a - b) <= (atol + rtol * abs(b)),
    pi=3.141592653589793,
    ndarray=object,
)
sys.modules.setdefault("numpy", np_stub)
np = np_stub

# Stub out dependencies not needed for opacity calculations
sys.modules.setdefault("amrex", types.ModuleType("amrex"))
sys.modules.setdefault("adios2", types.ModuleType("adios2"))
numba_stub = types.ModuleType("numba")
numba_stub.njit = lambda *a, **k: (lambda f: f)
numba_stub.prange = range
sys.modules.setdefault("numba", numba_stub)
scipy_interp = types.ModuleType("scipy.interpolate")
scipy_interp.RegularGridInterpolator = lambda *a, **k: None
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.interpolate", scipy_interp)
scipy_integrate = types.ModuleType("scipy.integrate")
scipy_integrate.solve_ivp = lambda *a, **k: None
sys.modules.setdefault("scipy.integrate", scipy_integrate)
models_stub = types.ModuleType("models")
models_stub.PhysicsModule = object
models_stub.SimulationState = object
sys.modules.setdefault("models", models_stub)
config_stub = types.ModuleType("config_schema")
config_stub.RadiationConfig = object
sys.modules.setdefault("config_schema", config_stub)


def _make_model(model: str, params: dict) -> RadiationModel:
    rm = RadiationModel.__new__(RadiationModel)
    rm.opacity_model = model
    rm.opacity_params = params
    return rm


def test_constant_opacity():
    rm = _make_model("constant", {"constant_opacity": 2.5})
    assert np.allclose(rm._compute_opacity(Te=0.0, ne=0.0, Z=0.0), 2.5)


def test_temperature_dependent_opacity():
    rm = _make_model("temperature_dependent", {"base": 1.0, "alpha": 0.5, "beta": 2.0})
    Te = 3.0
    expected = 1.0 + 0.5 * Te**2.0
    assert np.allclose(rm._compute_opacity(Te=Te, ne=0.0, Z=0.0), expected)
