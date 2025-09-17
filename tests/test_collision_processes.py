import numpy as np
import pytest
import sys
import types

sys.modules.setdefault(
    "h5py", types.SimpleNamespace(File=lambda *a, **k: (_ for _ in ()).throw(OSError()))
)
scipy_interp = types.SimpleNamespace(
    interp1d=lambda *a, **k: (lambda x: np.zeros_like(x)),
    RegularGridInterpolator=lambda *a, **k: (lambda x: np.zeros(1)),
)
sys.modules.setdefault("scipy", types.SimpleNamespace())
sys.modules.setdefault("scipy.interpolate", scipy_interp)
sys.modules.setdefault(
    "numba",
    types.SimpleNamespace(
        njit=lambda f=None, *a, **k: (
            lambda *args, **kwargs: f(*args, **kwargs) if f else None
        ),
        prange=range,
        cuda=types.SimpleNamespace(),
    ),
)
np.ndarray = np.Array
np.random = types.SimpleNamespace()

from dpf2.simulation.collision_model import (
    ElectronIonCollision,
    IonizationProcess,
    nu_ei_spitzer,
    kB,
    m_e,
    pi,
)
from dpf2.simulation.utils import SimulationState


class DummyFieldManager:
    def __init__(self, ne, Te, nn=None):
        self.ne = ne
        self.Te = Te
        self.nn = nn if nn is not None else 0

    def get_J(self):
        return np.zeros((3,) + self.ne.shape)


def make_state(species, fm):
    state = SimulationState((1, 1, 1), 1.0, 1.0, 1.0, (0, 0, 0), {})
    state.species = species
    state.field_manager = fm
    return state


def test_electron_ion_collision_velocity_reduction(monkeypatch):
    ne = np.array([1.0, 1.0])
    Te = np.array([1.0, 1.0])
    fm = DummyFieldManager(ne, Te)
    vel0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    species = {"e": {"q": -1.0, "vel": vel0.copy()}}
    state = make_state(species, fm)
    dt = 0.1
    collision = ElectronIonCollision()
    monkeypatch.setattr(
        "dpf2.simulation.collision_model.nu_ei_spitzer",
        lambda ne, Te: np.array([5.0, 5.0]),
    )
    collision.apply(state, dt)
    expected_vel = vel0 - np.array([5.0, 5.0])[:, None] * vel0 * dt
    assert np.allclose(state.species["e"]["vel"], expected_vel)


def test_ionization_process_uses_dt(monkeypatch):
    ne = 1e19
    Te = 1e3
    nn = 1e20
    fm = DummyFieldManager(ne, Te, nn)
    species = {"e": {"q": -1.0, "vel": np.zeros((1, 3))}}
    state = make_state(species, fm)
    process = IonizationProcess()
    sigma = 2e-20
    process.cross_section_data = lambda T: sigma
    captured = {}

    def fake_poisson(lam):
        captured["lam"] = lam
        return np.zeros_like(lam, dtype=int)

    monkeypatch.setattr(np.random, "poisson", fake_poisson, raising=False)
    dt = 0.1
    process.apply(state, dt)
    import math

    ion_rate = ne * sigma * math.sqrt(8 * kB * Te / (pi * m_e))
    expected = ion_rate * nn * dt
    assert np.allclose(captured["lam"], expected)
