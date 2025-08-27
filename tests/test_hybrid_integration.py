import os
import sys
import numpy as np
import pytest

import types
from contextlib import contextmanager

# Stub modules required by hybrid_controller
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

# Provide minimal stubs for utilities to avoid heavy dependencies.
utils_mod = types.ModuleType("dpf2.simulation.utils")
class FieldManager:
    def __init__(self, *args, **kwargs):
        pass
    def get_J(self):
        return 0.0
class SimulationState:
    def __init__(self, *args, **kwargs):
        self.density = kwargs.get("density")
        self.velocity = kwargs.get("velocity")
        self.pressure = kwargs.get("pressure")
        self.electron_temperature = kwargs.get("electron_temperature")
        self.ion_temperature = kwargs.get("ion_temperature")
        self.field_manager = kwargs.get("field_manager")
sys.modules["dpf2.simulation.utils"] = utils_mod
utils_mod.FieldManager = FieldManager
utils_mod.SimulationState = SimulationState

HybridController = pytest.importorskip("dpf2.simulation.hybrid_controller").HybridController


def test_hybrid_step_combines_fluid_and_pic():
    grid_shape = (2, 2, 2)
    bc = {
        'x_lo': 'periodic', 'x_hi': 'periodic',
        'y_lo': 'periodic', 'y_hi': 'periodic',
        'z_lo': 'periodic', 'z_hi': 'periodic'
    }
    fm = FieldManager(grid_shape, 1.0, 1.0, 1.0, (0.0, 0.0, 0.0), bc)

    density = np.ones(grid_shape)
    velocity = np.zeros(grid_shape + (3,))
    pressure = np.zeros(grid_shape)
    temperature = np.zeros(grid_shape)

    state = SimulationState(
        grid_shape, 1.0, 1.0, 1.0, (0.0, 0.0, 0.0), bc,
        density=density, velocity=velocity, pressure=pressure,
        electron_temperature=temperature, ion_temperature=temperature,
        field_manager=fm
    )

    class DummyFluid:
        def __init__(self):
            self.state = {'density': density}
            self.step_called = False
            self.energy_increments = []

        def get_total_energy(self):
            return 0.0

        def step(self, dt):
            self.step_called = True

        def increment_internal_energy(self, corr):
            self.energy_increments.append(corr)

    class DummyPIC:
        def __init__(self):
            self.step_calls = 0

        def step(self, fluid_data, dt, region, it):
            self.step_calls += 1
            rho = fluid_data['density']
            vel = fluid_data['velocity']
            momentum_density = rho[..., None] * vel + 0.1
            pressure_density = np.full_like(rho, 0.2)
            return {
                'momentum_density': momentum_density,
                'pressure_density': pressure_density,
            }

    class DummyCircuit:
        def __init__(self):
            self.step_calls = 0
        def step(self, current, back_emf, dt, feedback=None):
            self.step_calls += 1
            return current, 0.0

    class DummyRadiation:
        def __init__(self):
            self.apply_calls = 0

        def apply(self, state, dt):
            self.apply_calls += 1
            state.radiation_calls = getattr(state, 'radiation_calls', 0) + 1

    class DummySheath:
        def __init__(self):
            self.apply_calls = 0

        def apply(self, state, phi):
            self.apply_calls += 1

    cfg = type('cfg', (), {})()
    cfg.criteria = type('criteria', (), {
        'grad_thr': 0.0,
        'knud_thr': 0.0,
        'hall_thr': 0.0,
        'non_max_fac': 1.0,
    })()
    cfg.coupling = type('coupling', (), {
        'buffer_cells': 0,
        'filter_sigma': 0,
        'blend_width': 1,
        'max_iters': 1,
        'coupling_tol': 1e-3,
        'target_vol_frac': 0.5,
        'max_subcycles': 1,
    })()

    fluid = DummyFluid()
    pic = DummyPIC()
    circuit = DummyCircuit()
    radiation = DummyRadiation()
    sheath = DummySheath()
    controller = HybridController(cfg, fluid, pic, circuit, radiation, None, sheath, fm)

    region = (slice(0, 2), slice(0, 2), slice(0, 2))
    controller.hybrid_step(state, [region], dt=1.0)

    assert fluid.step_called
    assert pic.step_calls > 0
    assert circuit.step_calls == 1
    assert radiation.apply_calls == 1
    assert getattr(state, 'radiation_calls', 0) == 1
    assert np.allclose(state.velocity, 0.1)
    assert np.allclose(state.pressure, 0.2)
if not hasattr(np, "ndarray"):
    pytest.skip("requires numpy", allow_module_level=True)
