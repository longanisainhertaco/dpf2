import sys
import pathlib
from types import SimpleNamespace, ModuleType

# Use the lightweight numpy shim unless a real installation is available.
sys.modules.pop("numpy", None)
sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "venv/lib/python3.12/site-packages"
    ),
)
import numpy as np
import pydantic_stub

sys.modules["pydantic"] = pydantic_stub

# Provide a minimal numba stub if numba is unavailable
if "numba" not in sys.modules:
    numba_stub = ModuleType("numba")
    numba_stub.njit = lambda *a, **k: (lambda f: f)
    numba_stub.prange = range
    sys.modules["numba"] = numba_stub

# Stub out optional WarpX wrapper dependency
warpx_stub = ModuleType("dpf2.simulation.warpx_wrapper")


class _WarpXWrapper:
    def __init__(self, *a, **k):
        pass


warpx_stub.WarpXWrapper = _WarpXWrapper
sys.modules["dpf2.simulation.warpx_wrapper"] = warpx_stub

# Stub collision processes to avoid heavy dependencies
coll_stub = ModuleType("dpf2.simulation.collision_model")


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
sys.modules["dpf2.simulation.collision_model"] = coll_stub

from pathlib import Path
from dpf2.simulation.pic_solver import PICSolver
from dpf2.validation_suite import load_pinch_dataset


def make_config():
    return SimpleNamespace(
        grid_shape=(4, 4, 4),
        grid_spacing=(1.0, 1.0, 1.0),
        max_dt=None,
        electromag="yee",
        boundary_conditions={"x": "reflecting", "y": "reflecting", "z": "reflecting"},
        dt=0.1,
        use_warpx=False,
        unity_params={},
        vdf_bins=8,
        max_vel=1.0,
        subgrid_resolution=1,
        amr=False,
        density_threshold=1.0,
        enable_quantum=False,
        enable_radiation=False,
        enable_mesh_adaptivity=False,
    )


def make_field_manager():
    shape = (4, 4, 4)
    fm = SimpleNamespace()
    fm.E = np.zeros((3,) + shape)
    fm.B = np.zeros((3,) + shape)
    fm.J = np.ones((3,) + shape)
    fm.rho = np.zeros(shape)
    for i in range(shape[0]):
        fm.rho[i, :, :] = i  # gradient in x
    for j in range(shape[1]):
        fm.B[2, :, j, :] = j  # Bz gradient in y
    fm.get_E = lambda: fm.E
    fm.get_B = lambda: fm.B
    fm.get_J = lambda: fm.J
    fm.get_rho = lambda: fm.rho
    fm.update_E = lambda E: setattr(fm, "E", E)
    fm.update_B = lambda B: setattr(fm, "B", B)
    fm.update_J = lambda J: setattr(fm, "J", J)
    return fm


def test_lhdi_matches_mjolnir():
    bench = load_pinch_dataset(Path("data/benchmarks/LLNL_MJOLNIR"))
    _, voltage = bench["voltage"]
    expected = float(np.max(voltage))

    cfg = make_config()
    fm = make_field_manager()
    solver = PICSolver(cfg, fm)
    solver.collisions = []
    solver.set_lhdi_model(expected / np.sqrt(3))
    solver.pml_sigma_e = np.zeros(solver.nz)
    solver.pml_sigma_b = np.zeros(solver.nz)
    solver.solve_fields()

    assert np.isclose(solver.voltage_spikes[-1], expected)
    eta_field = solver.lhdi_model.compute_eta(
        np.abs(fm.rho), fm.B, (solver.dx, solver.dy, solver.dz)
    )
    assert np.allclose(eta_field, expected / np.sqrt(3))
