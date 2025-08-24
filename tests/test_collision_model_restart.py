import sys
import types
import pytest


def _raise(*args, **kwargs):
    raise OSError("file not found")


@pytest.fixture
def collision_model_classes(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace())
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

    from Simulation.collision_model import CollisionModel, CrossSectionData
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

