import numpy as np
import pytest
import sys
import types

# Stub optional heavy dependencies
sys.modules.setdefault("h5py", types.SimpleNamespace(File=lambda *a, **k: (_ for _ in ()).throw(OSError())))
pyevtk_stub = types.ModuleType("pyevtk")
hl_stub = types.ModuleType("pyevtk.hl")
hl_stub.imageToVTK = lambda *a, **k: None
pyevtk_stub.hl = hl_stub
sys.modules.setdefault("pyevtk", pyevtk_stub)
sys.modules.setdefault("pyevtk.hl", hl_stub)
scipy_stub = types.SimpleNamespace(
    constants=types.SimpleNamespace(c=1.0, m_n=1.0, m_e=1.0, mu_0=1.0, e=1.0, epsilon_0=1.0, k=1.0),
    interpolate=types.SimpleNamespace(interp1d=lambda *a, **k: None),
)
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.constants", scipy_stub.constants)
sys.modules.setdefault("scipy.interpolate", scipy_stub.interpolate)
# Minimal numba stub for collision_model
sys.modules.setdefault(
    "numba",
    types.SimpleNamespace(
        njit=lambda f=None, *a, **k: (lambda *args, **kwargs: f(*args, **kwargs) if f else None),
        prange=range,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    ),
)

# restore numpy random for local tests
np.random = types.SimpleNamespace(
    normal=lambda loc, scale, size: np.ones(size),
    poisson=lambda lam, size=None: 0,
)

from dpf2.simulation.collision_model import (
    FokkerPlanckOperator,
    AnisotropyRelaxation,
    CollisionalRadiativeNetwork,
)
from dpf2.simulation.gpu_diagnostics import GPUKineticEnergyDiagnostic
from dpf2.simulation.utils import SimulationState


class DummyFieldManager:
    def __init__(self, ne=None, Te=None, nn=None):
        self.ne = ne if ne is not None else np.array([0.0])
        self.Te = Te if Te is not None else np.array([0.0])
        self.nn = nn if nn is not None else np.array([0.0])

    def get_J(self):
        return np.zeros((3,) + self.ne.shape)


def make_state(species, fm, rad=None):
    state = SimulationState((1, 1, 1), 1.0, 1.0, 1.0, (0, 0, 0), {})
    state.species = species
    state.field_manager = fm
    if rad is not None:
        state.radiation = rad
    return state


def test_fokker_planck_diffusion(monkeypatch):
    vel = np.zeros((2, 3))
    species = {"e": {"q": -1.0, "vel": vel}}
    state = make_state(species, DummyFieldManager())
    op = FokkerPlanckOperator(diffusion_coeff=1.0)
    op.apply(state, 0.5)
    expected_val = (2.0 * 1.0 * 0.5) ** 0.5
    assert state.species["e"]["vel"].data == [[expected_val]*3, [expected_val]*3]


def test_anisotropy_relaxation_reduces_ratio():
    vel = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    species = {"i": {"q": 1.0, "vel": vel}}
    state = make_state(species, DummyFieldManager())
    def _var(a):
        data = a.data if hasattr(a, "data") else a
        n = len(data)
        sums = [0.0, 0.0, 0.0]
        sqs = [0.0, 0.0, 0.0]
        for row in data:
            for j in range(3):
                val = row[j]
                sums[j] += val
                sqs[j] += val * val
        return [sqs[j]/n - (sums[j]/n)**2 for j in range(3)]
    var0 = _var(vel)
    ratio0 = max(var0) / (min(var0) + 1e-12)
    op = AnisotropyRelaxation(rate=1.0)
    op.apply(state, 1.0)
    var1 = _var(state.species["i"]["vel"])
    ratio1 = max(var1) / (min(var1) + 1e-12)
    assert ratio1 < ratio0


def test_collisional_radiative_network_updates_populations():
    rad = {"populations": [1.0, 0.0]}
    state = make_state({}, DummyFieldManager(), rad)
    net = CollisionalRadiativeNetwork(levels=["g", "e"], coll_rates={(0, 1): 0.5}, rad_rates={(1, 0): 0.1})
    net.apply(state, 1.0)
    pops = state.radiation["populations"]
    assert all(abs(p - e) < 1e-12 for p, e in zip(pops, [0.55, 0.45]))


def test_gpu_kinetic_energy_diagnostic(monkeypatch):
    fm = DummyFieldManager()
    vel = np.array([[1.0, 0.0, 0.0]])
    species = {"e": {"q": -1.0, "m": 1.0, "vel": vel}}
    state = make_state(species, fm)
    diag = GPUKineticEnergyDiagnostic("e")
    diag.record(state)
    assert abs(diag.data[0]["ke"] - 0.5) < 1e-12
