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

from Simulation.collision_model import CollisionModel


def test_checkpoint_restart_roundtrip():
    cm1 = CollisionModel({})
    ion_cs = types.SimpleNamespace(name='ion')
    dd_cs = types.SimpleNamespace(name='dd')
    crn = types.SimpleNamespace(rates={'val': 1})
    cm1.ionization_cross_section = ion_cs
    cm1.dd_fusion_cross_section = dd_cs
    cm1.crn = crn

    data = cm1.checkpoint()

    cm2 = CollisionModel({})
    assert cm2.ionization_cross_section is not ion_cs
    assert cm2.dd_fusion_cross_section is not dd_cs
    assert cm2.crn is not crn

    cm2.restart(data)

    assert cm2.ionization_cross_section is ion_cs
    assert cm2.dd_fusion_cross_section is dd_cs
    assert cm2.crn is crn
