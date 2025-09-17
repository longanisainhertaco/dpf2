import sys
import types
import math
import numpy as np
import pytest
from contextlib import contextmanager

# ---- Stub modules required for hybrid_controller import ----
caliper = types.ModuleType("caliper")


@contextmanager
def annotate(name):
    yield


caliper.annotate = annotate
sys.modules["caliper"] = caliper

sys.modules["fluid_solver_high_order"] = types.ModuleType("fluid_solver_high_order")
sys.modules["fluid_solver_high_order"].FluidSolverHighOrder = object
sys.modules["warpx_wrapper"] = types.ModuleType("warpx_wrapper")
sys.modules["warpx_wrapper"].WarpXWrapper = object
sys.modules["radiation_model"] = types.ModuleType("radiation_model")
sys.modules["radiation_model"].RadiationModel = object
sys.modules["collision_model"] = types.ModuleType("collision_model")
sys.modules["collision_model"].CollisionModel = object
sys.modules["sheath_model"] = types.ModuleType("sheath_model")
sys.modules["sheath_model"].PlasmaSheathFormation = object
sys.modules["config_schema"] = types.ModuleType("config_schema")
sys.modules["config_schema"].HybridConfig = type("HybridConfig", (), {})
sys.modules["models"] = types.ModuleType("models")
sys.modules["models"].PhysicsModule = type("PhysicsModule", (), {})
sys.modules["models"].SimulationState = type("SimulationState", (), {})

sys.modules["dpf2.simulation.fluid_solver_high_order"] = sys.modules[
    "fluid_solver_high_order"
]
sys.modules["dpf2.simulation.warpx_wrapper"] = sys.modules["warpx_wrapper"]
sys.modules["dpf2.simulation.radiation_model"] = sys.modules["radiation_model"]
sys.modules["dpf2.simulation.collision_model"] = sys.modules["collision_model"]
sys.modules["dpf2.simulation.sheath_model"] = sys.modules["sheath_model"]
sys.modules["dpf2.simulation.config_schema"] = sys.modules["config_schema"]
sys.modules["dpf2.simulation.models"] = sys.modules["models"]

scipy = types.ModuleType("scipy")
nd = types.SimpleNamespace(
    gaussian_filter=lambda x, sigma: x,
    label=lambda mask: (mask, 0),
)
scipy.ndimage = nd
sys.modules["scipy"] = scipy
sys.modules["scipy.ndimage"] = nd

numba = types.ModuleType("numba")


def _njit(*args, **kwargs):
    def wrap(f):
        return f

    return wrap


numba.njit = _njit
numba.prange = range
sys.modules["numba"] = numba

utils_mod = types.ModuleType("dpf2.simulation.utils")


class FieldManager:
    def __init__(self, *args, **kwargs):
        pass

    def get_J(self):
        return 0.0

    def get_E(self):
        return np.zeros((1, 1, 1, 3))

    def get_B(self):
        return np.zeros((1, 1, 1, 3))


class SimulationState:
    def __init__(self, *args, **kwargs):
        self.density = kwargs.get("density")
        self.velocity = kwargs.get("velocity")
        self.pressure = kwargs.get("pressure")
        self.electron_temperature = kwargs.get("electron_temperature")
        self.ion_temperature = kwargs.get("ion_temperature")
        self.field_manager = kwargs.get("field_manager")
        self.dx = self.dy = self.dz = 1.0


sys.modules["dpf2.simulation.utils"] = utils_mod
utils_mod.FieldManager = FieldManager
utils_mod.SimulationState = SimulationState

_orig_zeros = np.zeros


def _zeros(shape, dtype=None):
    return _orig_zeros(shape)


np.zeros = _zeros

np.zeros_like = lambda arr: np.zeros(arr.shape)
np.ones_like = lambda arr: np.ones(arr.shape)
np.bool_ = bool


def _norm(v, axis=None):
    if axis is None:
        return math.sqrt(sum(x * x for x in v))
    shape = v.shape[:axis] + v.shape[axis + 1 :]
    return np.zeros(shape)


np.linalg = types.SimpleNamespace(norm=_norm)

HybridController = pytest.importorskip(
    "dpf2.simulation.hybrid_controller"
).HybridController
from dpf2.core.bases import CouplingState


# ---- Helper Dummies ----
class DummyFluid:
    def __init__(self):
        self.state = {"density": None}
        self.energy_increments = []

    def get_total_energy(self):
        return 0.0

    def step(self, dt):
        pass

    def increment_internal_energy(self, corr):
        self.energy_increments.append(corr)


class DummyPIC:
    def step(self, fluid_data, dt, region, it):
        return {
            "momentum_density": fluid_data["density"][..., None]
            * fluid_data["velocity"],
            "pressure_density": fluid_data["density"],
        }


class DummyCircuit:
    def get_voltage(self):
        return 0.0

    def step(self, *args, **kwargs):
        pass


class DummyRadiation:
    def apply(self, state, dt):
        pass

    def finalize(self):
        pass


class DummySheath:
    def apply(self, state, dt):
        pass


# ---- Tests ----


def _controller(collision_model):
    cfg = type("cfg", (), {})()
    cfg.criteria = type(
        "criteria",
        (),
        {
            "grad_thr": -1.0,
            "knud_thr": 1e9,
            "hall_thr": 1e9,
            "non_max_fac": 1.0,
        },
    )()
    cfg.coupling = type(
        "coupling",
        (),
        {
            "buffer_cells": 0,
            "filter_sigma": 0,
            "blend_width": 1,
            "max_iters": 1,
            "coupling_tol": 1e-3,
            "target_vol_frac": 0.5,
            "max_subcycles": 1,
        },
    )()
    fm = FieldManager()
    return HybridController(
        cfg,
        DummyFluid(),
        DummyPIC(),
        DummyCircuit(),
        DummyRadiation(),
        collision_model,
        DummySheath(),
        fm,
    )


def test_collision_frequency_includes_extra_models():
    class Coll:
        def nu_ei_spitzer(self, ne, Te):
            return ne * 0 + 1

        def nu_molecular(self, ne, Te):
            return ne * 0 + 2

        def nu_dust(self, ne, Te):
            return ne * 0 + 3

    ctrl = _controller(Coll())
    state = SimulationState(
        density=np.ones((1, 1, 1)),
        electron_temperature=np.ones((1, 1, 1)),
        velocity=np.zeros((1, 1, 1, 3)),
        pressure=np.zeros((1, 1, 1)),
        ion_temperature=np.zeros((1, 1, 1)),
        field_manager=FieldManager(),
    )
    nu = ctrl.compute_collision_frequency(state)
    assert nu[0][0][0] == 6.0


def test_collision_frequency_error_handling():
    class Coll:
        def nu_ei_spitzer(self, ne, Te):
            raise RuntimeError("boom")

    ctrl = _controller(Coll())
    state = SimulationState(
        density=np.ones((1, 1, 1)),
        electron_temperature=np.ones((1, 1, 1)),
        velocity=np.zeros((1, 1, 1, 3)),
        pressure=np.zeros((1, 1, 1)),
        ion_temperature=np.zeros((1, 1, 1)),
        field_manager=FieldManager(),
    )
    nu = ctrl.compute_collision_frequency(state)
    assert nu[0][0][0] == 0.0


def test_non_lte_and_time_update():
    class Coll:
        def nu_ei_spitzer(self, ne, Te):
            return np.zeros_like(ne)

    ctrl = _controller(Coll())
    state = SimulationState(
        density=np.ones((1, 1, 1)),
        electron_temperature=2 * np.ones((1, 1, 1)),
        ion_temperature=np.zeros((1, 1, 1)),
        velocity=np.zeros((1, 1, 1, 3)),
        pressure=np.zeros((1, 1, 1)),
        field_manager=FieldManager(),
    )
    import dpf2.simulation.hybrid_controller as hc

    def _mask(*a, **k):
        m = np.zeros((1, 1, 1))
        m.sum = lambda: 0
        m.size = 1
        return m

    hc.compute_transition_mask = _mask
    ctrl.apply(state, dt=1.0)
    assert ctrl.time == pytest.approx(1.0)
    assert state.electron_temperature[0][0][0] == pytest.approx(1.8)
    assert state.ion_temperature[0][0][0] == pytest.approx(0.2)


def test_relativistic_and_quantum_corrections():
    class Coll:
        def nu_ei_spitzer(self, ne, Te):
            return np.zeros_like(ne)

    ctrl = _controller(Coll())
    v0 = 0.9 * 299792458.0
    state = SimulationState(
        density=1e30 * np.ones((1, 1, 1)),
        electron_temperature=np.ones((1, 1, 1)),
        ion_temperature=np.ones((1, 1, 1)),
        velocity=v0 * np.ones((1, 1, 1, 3)),
        pressure=np.zeros((1, 1, 1)),
        field_manager=FieldManager(),
    )
    ctrl.fluid_only_step(state, dt=1.0)
    speed = math.sqrt(sum(v * v for v in state.velocity[0][0][0]))
    assert speed < v0
    assert state.pressure[0][0][0] > 0
