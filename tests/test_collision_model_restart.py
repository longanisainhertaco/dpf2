import types
import sys

# Stub external dependencies required by collision_model
sys.modules['numpy'] = types.SimpleNamespace()

def _raise(*args, **kwargs):
    raise OSError("file not found")

sys.modules['h5py'] = types.SimpleNamespace(File=_raise)

interp_stub = types.SimpleNamespace(
    interp1d=lambda *a, **k: (lambda x: 0.0),
    RegularGridInterpolator=lambda *a, **k: (lambda x: 0.0),
)
sys.modules['scipy'] = types.SimpleNamespace()
sys.modules['scipy.interpolate'] = interp_stub

numba_stub = types.SimpleNamespace(
    njit=lambda f=None, *a, **k: (lambda *args, **kwargs: f(*args, **kwargs) if f else None),
    prange=range,
    cuda=types.SimpleNamespace(),
)
sys.modules['numba'] = numba_stub

# Stub models module used in collision_model
models_stub = types.SimpleNamespace(
    PhysicsModule=object,
    SimulationState=object,
)
sys.modules['models'] = models_stub

from Simulation.collision_model import CollisionModel, CrossSectionData


def test_checkpoint_restart_roundtrip():
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


def test_checkpoint_restart_idempotent():
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
