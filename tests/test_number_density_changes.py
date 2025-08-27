import numpy as np
import types
import sys


class DummyFieldManager:
    def __init__(self, ne, Te, nn, ni):
        self.ne = ne
        self.Te = Te
        self.nn = nn
        self.ni = ni

    def get_J(self):
        return np.zeros((3,))


class DeterministicRandom:
    def poisson(self, lam):
        try:
            _ = len(lam)  # array-like
            return np.zeros_like(lam, dtype=int) + 1
        except Exception:
            return 1

    def normal(self, loc=0.0, scale=1.0, size=None):
        return np.zeros(size)

    def random(self, size=None):
        return np.zeros(size)

    def randint(self, low, high=None):
        return 0


def test_number_density_changes_over_time(monkeypatch):
    # Stub heavy dependencies before import
    monkeypatch.setitem(sys.modules, "h5py", types.SimpleNamespace(File=lambda *a, **k: (_ for _ in ()).throw(OSError())))
    scipy_interp = types.SimpleNamespace(
        interp1d=lambda *a, **k: (lambda x: np.zeros_like(x)),
        RegularGridInterpolator=lambda *a, **k: (lambda x: np.zeros(1)),
    )
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "scipy.interpolate", scipy_interp)
    numba_stub = types.SimpleNamespace(
        njit=lambda f=None, *a, **k: (lambda *args, **kwargs: f(*args, **kwargs) if f else None),
        prange=range,
        cuda=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "numba", numba_stub)

    from dpf2.simulation.collision_model import IonizationProcess, RecombinationProcess, e_charge, m_e, m_p
    from dpf2.simulation.utils import SimulationState

    dr = DeterministicRandom()
    monkeypatch.setattr(np, "random", dr, raising=False)

    fm = DummyFieldManager(ne=1e18, Te=1e3, nn=1e20, ni=1e18)
    state = SimulationState((1, 1, 1), 1.0, 1.0, 1.0, (0.0, 0.0, 0.0), {})
    state.field_manager = fm
    state.species = {
        "e": {"q": -e_charge, "m": m_e, "pos": np.zeros((0, 3)), "vel": np.zeros((0, 3))},
        "ion": {"q": e_charge, "m": m_p, "pos": np.zeros((0, 3)), "vel": np.zeros((0, 3))},
    }

    ion = IonizationProcess()
    ion.cross_section_data = lambda T: 1e-20
    for _ in range(2):
        ion.apply(state, 1.0)
    assert state.species["e"]["pos"].shape[0] == 2
    assert state.species["ion"]["pos"].shape[0] == 2

    rec = RecombinationProcess(recombination_rate=1e-20)
    rec.apply(state, 1.0)
    assert state.species["e"]["pos"].shape[0] == 1
    assert state.species["ion"]["pos"].shape[0] == 1
