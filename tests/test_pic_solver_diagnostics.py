import sys
import pathlib
sys.modules.pop('numpy', None)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'venv/lib/python3.12/site-packages'))
import numpy as np
from types import SimpleNamespace, ModuleType
import pydantic_stub
sys.modules['pydantic'] = pydantic_stub

# Provide a minimal numba stub if numba is unavailable
if 'numba' not in sys.modules:
    numba_stub = ModuleType('numba')
    numba_stub.njit = lambda *a, **k: (lambda f: f)
    numba_stub.prange = range
    sys.modules['numba'] = numba_stub

# Stub out optional WarpX wrapper dependency
warpx_stub = ModuleType('dpf2.simulation.warpx_wrapper')
class _WarpXWrapper:
    def __init__(self, *a, **k):
        pass
warpx_stub.WarpXWrapper = _WarpXWrapper
sys.modules['dpf2.simulation.warpx_wrapper'] = warpx_stub

# Stub collision processes to avoid heavy dependencies
coll_stub = ModuleType('dpf2.simulation.collision_model')
class _CollisionProcess:
    def __init__(self, *a, **k):
        pass
    def apply(self, solver, dt):
        pass
coll_stub.CollisionProcess = _CollisionProcess
coll_stub.BetheBlochStopping = _CollisionProcess
coll_stub.ElectronIonCollision = _CollisionProcess
coll_stub.ElectronNeutralCollision = _CollisionProcess
coll_stub.IonizationProcess = _CollisionProcess
coll_stub.RecombinationProcess = _CollisionProcess
sys.modules['dpf2.simulation.collision_model'] = coll_stub

from dpf2.simulation.pic_solver import PICSolver


def make_config(bcs, dt=1.0, quantum=False, radiation=False, adapt=False):
    return SimpleNamespace(
        grid_shape=(4, 4, 4),
        grid_spacing=(1.0, 1.0, 1.0),
        max_dt=None,
        electromag='yee',
        boundary_conditions=bcs,
        dt=dt,
        use_warpx=False,
        unity_params={},
        vdf_bins=8,
        max_vel=1.0,
        subgrid_resolution=1,
        amr=False,
        density_threshold=1.0,
        enable_quantum=quantum,
        enable_radiation=radiation,
        enable_mesh_adaptivity=adapt,
    )


def make_field_manager():
    shape = (4, 4, 4)
    fm = SimpleNamespace()
    fm.E = np.zeros((3,) + shape)
    fm.B = np.zeros((3,) + shape)
    fm.J = np.zeros((3,) + shape)
    fm.rho = np.zeros(shape)
    fm.get_E = lambda: fm.E
    fm.get_B = lambda: fm.B
    fm.get_J = lambda: fm.J
    fm.get_rho = lambda: fm.rho
    fm.update_J = lambda J: None
    return fm


def test_energy_spectra_phase_space():
    cfg = make_config({'x': 'reflecting', 'y': 'reflecting', 'z': 'reflecting'}, dt=0.1)
    fm = make_field_manager()
    solver = PICSolver(cfg, fm)
    solver.collisions = []
    positions = np.zeros((10, 3))
    velocities = np.random.randn(10, 3)
    solver.add_species('e', -1.0, 1.0, positions, velocities)
    edges, hist = solver.compute_energy_spectra('e', bins=5)
    assert hist.sum() == 10
    xedges, vedges, H = solver.compute_phase_space('e', bins=5)
    assert H.shape == (5, 5)


def test_boundary_conditions():
    fm = make_field_manager()
    # Periodic in x
    cfg = make_config({'x': 'periodic', 'y': 'reflecting', 'z': 'reflecting'}, dt=1.0)
    solver = PICSolver(cfg, fm)
    solver.add_species('e', -1.0, 1.0, np.array([[3.9, 0.0, 0.0]]), np.array([[0.5, 0.0, 0.0]]))
    E = np.zeros_like(fm.E)
    B = np.zeros_like(fm.B)
    spc = solver.species['e']
    solver.boris_push_numba(spc['pos'], spc['vel'], spc['q'], spc['m'], solver.dt,
                            E, B, solver.origin, (solver.dx, solver.dy, solver.dz),
                            solver.bc_codes, (solver.nx, solver.ny, solver.nz))
    x = solver.species['e']['pos'][0, 0]
    assert np.isclose(x, 0.4, atol=1e-2)

    # Reflecting in x
    cfg = make_config({'x': 'reflecting', 'y': 'reflecting', 'z': 'reflecting'}, dt=1.0)
    solver = PICSolver(cfg, fm)
    solver.add_species('e', -1.0, 1.0, np.array([[3.9, 0.0, 0.0]]), np.array([[0.5, 0.0, 0.0]]))
    spc = solver.species['e']
    solver.boris_push_numba(spc['pos'], spc['vel'], spc['q'], spc['m'], solver.dt,
                            E, B, solver.origin, (solver.dx, solver.dy, solver.dz),
                            solver.bc_codes, (solver.nx, solver.ny, solver.nz))
    x = solver.species['e']['pos'][0, 0]
    v = solver.species['e']['vel'][0, 0]
    assert np.isclose(x, 3.6, atol=1e-2)
    assert np.isclose(v, -0.5, atol=1e-2)


def test_model_hooks_called():
    fm = make_field_manager()
    cfg = make_config({'x': 'reflecting', 'y': 'reflecting', 'z': 'reflecting'},
                      dt=0.1, quantum=True, radiation=True, adapt=True)
    solver = PICSolver(cfg, fm)
    solver.add_species('e', -1.0, 1.0, np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]))
    calls = []
    solver.set_quantum_model(lambda s: calls.append('q'))
    solver.set_radiation_model(lambda s: calls.append('r'))
    solver.set_mesh_adapter(lambda s: calls.append('m'))
    if solver.quantum_model:
        solver.quantum_model()
    if solver.radiation_model:
        solver.radiation_model()
    if solver.mesh_adapter:
        solver.mesh_adapter()
    assert set(calls) == {'q', 'r', 'm'}
