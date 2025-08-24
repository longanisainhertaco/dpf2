"""Regression tests for :mod:`CollisionModel` checkpoint/restart.

These tests operate in a heavily mocked environment so that they run quickly
and do not require any of the optional heavy dependencies present in the full
project.  Only the logic around serialising and deserialising the collision
model state is exercised.
"""

import sys
import types
import pytest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _raise(*args, **kwargs):
    raise OSError("file not found")


class RandomStub:
    """Minimal stub to emulate numpy.random for tests."""

    def __init__(self):
        import random
        self._rng = random.Random()

    def get_state(self):
        return self._rng.getstate()

    def set_state(self, state):
        self._rng.setstate(state)

    def rand(self):
        return self._rng.random()

    random = rand

    def seed(self, seed=None):
        self._rng.seed(seed)


@pytest.fixture
def collision_model_classes(monkeypatch):
    numpy_stub = types.SimpleNamespace(
        random=RandomStub(),
        array=lambda x: x,
        asarray=lambda x: x,
    )
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
    monkeypatch.setitem(sys.modules, "h5py", types.SimpleNamespace(File=_raise))

    interp_stub = types.SimpleNamespace(
        interp1d=lambda *a, **k: (lambda x: 0.0),
        RegularGridInterpolator=lambda *a, **k: (lambda x: 0.0),
    )
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "scipy.interpolate", interp_stub)

    numba_stub = types.SimpleNamespace(
        njit=lambda f=None, *a, **k: (lambda *args, **kwargs: f(*args, **kwargs) if f else None),
        prange=range,
        cuda=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "numba", numba_stub)

    models_stub = types.SimpleNamespace(
        PhysicsModule=object,
        SimulationState=object,
    )
    monkeypatch.setitem(sys.modules, "models", models_stub)

    dpf2_pkg = types.ModuleType("dpf2")
    dpf2_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "dpf2")]
    sys.modules["dpf2"] = dpf2_pkg

    from dpf2.simulation.collision_model import CollisionModel, CrossSectionData

    return CollisionModel, CrossSectionData


def test_checkpoint_restart_roundtrip(collision_model_classes):
    CollisionModel, CrossSectionData = collision_model_classes
    cm1 = CollisionModel({})
    ion_cs = CrossSectionData.from_dict({'energy': [1, 2], 'cross_section': [3, 4]})
    dd_cs = CrossSectionData.from_dict({'energy': [5, 6], 'cross_section': [7, 8]})
    crn = types.SimpleNamespace(rates={'val': 1})
    cm1.ionization_cross_section = ion_cs
    cm1.dd_fusion_cross_section = dd_cs
    cm1.crn = crn
    cm1.accumulators['steps'] = 10
    cm1.caches['nu_ei'] = [0.1, 0.2]

    data = cm1.checkpoint()

    cm2 = CollisionModel({})
    cm2.restart(data)

    assert cm2.crn is crn
    assert list(cm2.ionization_cross_section.energy) == [1, 2]
    assert list(cm2.dd_fusion_cross_section.cross_section) == [7, 8]
    assert cm2.accumulators == cm1.accumulators
    assert cm2.caches == cm1.caches


def test_checkpoint_restart_idempotent(collision_model_classes):
    CollisionModel, CrossSectionData = collision_model_classes
    cm1 = CollisionModel({})
    ion_cs = CrossSectionData.from_dict({'energy': [1], 'cross_section': [2]})
    dd_cs = CrossSectionData.from_dict({'energy': [3], 'cross_section': [4]})
    cm1.ionization_cross_section = ion_cs
    cm1.dd_fusion_cross_section = dd_cs
    cm1.accumulators['steps'] = 5
    cm1.caches['nu_ei'] = [0.3]

    data = cm1.checkpoint()

    cm2 = CollisionModel({})
    cm2.restart(data)

    assert cm2.checkpoint() == data


def test_restart_restores_random_state(collision_model_classes):
    CollisionModel, _ = collision_model_classes
    cm_module = sys.modules[CollisionModel.__module__]
    np = cm_module.np
    np.random.seed(123)
    cm = CollisionModel({})
    data = cm.checkpoint()
    expected = np.random.rand()
    np.random.rand()
    cm.restart(data)
    assert np.random.rand() == expected


def test_checkpoint_restart_reproduces_behavior(collision_model_classes):
    CollisionModel, CrossSectionData = collision_model_classes
    cm = CollisionModel({})
    ion_cs = CrossSectionData.from_dict({'energy': [1, 2], 'cross_section': [3, 4]})
    cm.ionization_cross_section = ion_cs
    energy_before = list(cm.ionization_cross_section.energy)
    xs_before = list(cm.ionization_cross_section.cross_section)
    data = cm.checkpoint()

    # Modify cross-section to ensure stored data is used on restart
    cm.ionization_cross_section = CrossSectionData.from_dict({'energy': [1, 2], 'cross_section': [30, 40]})
    assert list(cm.ionization_cross_section.cross_section) != xs_before

    cm.restart(data)
    assert list(cm.ionization_cross_section.energy) == energy_before
    assert list(cm.ionization_cross_section.cross_section) == xs_before


def test_checkpoint_restart_identical_evolution(collision_model_classes):
    """After restarting from a checkpoint the model should evolve identically."""
    CollisionModel, _ = collision_model_classes
    cm_module = sys.modules[CollisionModel.__module__]
    np = cm_module.np

    cm1 = CollisionModel({})

    # Capture a checkpoint of the initial state
    import copy
    data = copy.deepcopy(cm1.checkpoint())

    # "Evolve" the model once: update an accumulator and draw a random number
    cm1.accumulators['steps'] = cm1.accumulators.get('steps', 0) + 1
    rand1 = np.random.rand()
    cm1.caches['rand'] = rand1
    evolved1 = cm1.checkpoint()

    # Restart from the original checkpoint and perform the same evolution
    cm2 = CollisionModel({})
    cm2.restart(data)
    cm2.accumulators['steps'] = cm2.accumulators.get('steps', 0) + 1
    rand2 = np.random.rand()
    cm2.caches['rand'] = rand2
    evolved2 = cm2.checkpoint()

    # The random draw and the resulting state after evolution must match
    assert rand1 == rand2
    assert evolved1 == evolved2

