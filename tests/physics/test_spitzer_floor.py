from types import ModuleType, SimpleNamespace
import sys
import numpy as np
import pytest

if 'numba' not in sys.modules:  # provide minimal numba stub
    numba_stub = ModuleType('numba')
    numba_stub.njit = lambda *a, **k: (lambda f: f)
    numba_stub.prange = range
    sys.modules['numba'] = numba_stub

warpx_stub = ModuleType('dpf2.simulation.warpx_wrapper')
class _WarpXWrapper:
    def __init__(self, *a, **k):
        pass
warpx_stub.WarpXWrapper = _WarpXWrapper
sys.modules['dpf2.simulation.warpx_wrapper'] = warpx_stub

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

from dpf2.hall_mhd_solver import HallMHDSolver, MHDState
from dpf2.simulation.pic_solver import PICSolver


# --- Helper constructors for PIC solver ---

def make_config():
    return SimpleNamespace(
        grid_shape=(2, 2, 2),
        grid_spacing=(1.0, 1.0, 1.0),
        max_dt=None,
        electromag='yee',
        boundary_conditions={'x': 'reflecting', 'y': 'reflecting', 'z': 'reflecting'},
        dt=0.1,
        use_warpx=False,
        unity_params={},
        vdf_bins=8,
        max_vel=1.0,
        subgrid_resolution=1,
        amr=False,
        density_threshold=1.0,
        electron_temperature=1.0,  # ensure sizable Spitzer floor
        enable_quantum=False,
        enable_radiation=False,
        enable_mesh_adaptivity=False,
    )


def make_field_manager():
    shape = (2, 2, 2)
    fm = SimpleNamespace()
    fm.E = np.zeros((3,) + shape)
    fm.B = np.zeros((3,) + shape)
    fm.J = np.zeros((3,) + shape)
    fm.rho = np.ones(shape)
    fm.get_E = lambda: fm.E
    fm.get_B = lambda: fm.B
    fm.get_J = lambda: fm.J
    fm.get_rho = lambda: fm.rho
    fm.update_E = lambda E: setattr(fm, 'E', E)
    fm.update_B = lambda B: setattr(fm, 'B', B)
    fm.update_J = lambda J: setattr(fm, 'J', J)
    return fm


# --- Tests ---

def test_hall_mhd_aborts_when_below_floor():
    shape = (2, 2, 2)
    rho = np.ones(shape)
    mom = np.zeros(shape + (3,))
    B = np.zeros(shape + (3,))
    energy = np.ones(shape)
    state = MHDState(rho=rho, mom=mom, energy=energy, B=B)

    def zero_resistivity(J):
        return np.zeros(J.shape[:-1])

    solver = HallMHDSolver(anomalous_resistivity=zero_resistivity)
    with pytest.raises(RuntimeError, match="Spitzer floor"):
        solver.step(state, 0.1)


def test_pic_aborts_when_below_floor():
    cfg = make_config()
    fm = make_field_manager()
    solver = PICSolver(cfg, fm)
    solver.set_lhdi_model(0.0)
    with pytest.raises(RuntimeError, match="Spitzer floor"):
        solver.solve_fields()
