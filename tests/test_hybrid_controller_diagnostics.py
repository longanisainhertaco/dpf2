import sys
import types
import numpy as np
import pytest

# Stub out SciPy requirement before importing module
scipy_stub = types.ModuleType("scipy")
scipy_stub.__path__ = []
scipy_stub.ndimage = types.SimpleNamespace(gaussian_filter=lambda a, sigma: a,
                                           label=lambda m: (np.zeros_like(m), 0))
scipy_stub.constants = types.SimpleNamespace()
scipy_stub.interpolate = types.SimpleNamespace(interp1d=lambda *a, **k: None,
                                               RegularGridInterpolator=lambda *a, **k: None)
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.ndimage", scipy_stub.ndimage)
sys.modules.setdefault("scipy.constants", scipy_stub.constants)
sys.modules.setdefault("scipy.interpolate", scipy_stub.interpolate)

# Stub out numba before importing module
numba_stub = types.ModuleType("numba")
numba_stub.njit = lambda *a, **k: (lambda f: f)
numba_stub.prange = range
numba_stub.cuda = types.SimpleNamespace()
sys.modules.setdefault("numba", numba_stub)
sys.modules.setdefault("numba.cuda", numba_stub.cuda)

sys.modules.setdefault("h5py", types.ModuleType("h5py"))

sys.modules.setdefault("picmi", types.ModuleType("picmi"))
sys.modules.setdefault("pywarpx", types.ModuleType("pywarpx"))

import numpy as np
np.linalg = types.SimpleNamespace(norm=lambda arr, axis=None: 0.0)
np.any = lambda a: False
np.isnan = lambda a: False

# Patch compute_transition_mask to avoid numba cost
from dpf2.simulation import hybrid_controller as hc_mod
def _mask(*a, **k):
    m = np.zeros((1,1,1))
    m.sum = lambda: 0
    m.size = 1
    return m

hc_mod.compute_transition_mask = _mask

class DummyState:
    def __init__(self):
        self.density = np.ones((1,1,1))
        self.velocity = np.zeros((1,1,1,3))
        self.pressure = np.ones((1,1,1))
        self.electron_temperature = np.ones((1,1,1))
        self.ion_temperature = np.ones((1,1,1))
        self.dx = self.dy = self.dz = 1.0
        self.field_manager = types.SimpleNamespace(get_B=lambda: np.zeros((1,1,1,3)))

class DummyFluid:
    def __init__(self):
        self.energy = 1.0
    def step(self, dt):
        self.energy += 0.1
    def get_total_energy(self):
        return self.energy
    def increment_internal_energy(self, corr):
        self.energy += corr

class DummyPIC:
    def __init__(self):
        self.energy = 0.5
    def get_total_energy(self):
        return self.energy

class DummyModule:
    def step(self, *a, **k):
        pass
    def apply(self, *a, **k):
        pass
    def get_voltage(self):
        return 0.0

config = types.SimpleNamespace(
    criteria=types.SimpleNamespace(grad_thr=0.1, knud_thr=0.1, hall_thr=0.1, non_max_fac=1.0),
    coupling=types.SimpleNamespace(buffer_cells=1, filter_sigma=1, blend_width=1, max_iters=1,
                                   coupling_tol=1.0, target_vol_frac=0.5, max_subcycles=1)
)


def test_diagnostic_histories(monkeypatch):
    hc = hc_mod.HybridController(config, DummyFluid(), DummyPIC(), DummyModule(), DummyModule(), DummyModule(), DummyModule(), types.SimpleNamespace(get_J=lambda:0.0))
    monkeypatch.setattr(hc, 'apply_boundary_conditions', lambda state, dt: None)
    monkeypatch.setattr(hc, 'compute_collision_frequency', lambda state: np.zeros((1,1,1)))
    monkeypatch.setattr(hc, 'fluid_only_step', lambda state, dt: hc.fluid.step(dt))

    state = DummyState()
    hc.apply(state, 0.1)

    assert hc.transition_history[-1]['vol_frac'] == 0.0
    assert hc.energy_history[-1]['fluid'] == pytest.approx(hc.fluid.get_total_energy())
    assert hc.energy_history[-1]['pic'] == pytest.approx(hc.pic.get_total_energy())
