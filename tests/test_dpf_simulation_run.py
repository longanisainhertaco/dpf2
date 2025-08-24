import pytest
from types import SimpleNamespace, ModuleType
from abc import ABC

class DummySolver:
    gamma = 1.4
    def __init__(self):
        self.dt_calls = 0
        self.steps = []
    def step(self, dt):
        self.steps.append(dt)
    def compute_optimal_dt(self):
        self.dt_calls += 1
        return 0.6

class DummyEOS:
    pass

class DummyCircuit:
    def __init__(self, collision_model=None, field_manager=None, **kwargs):
        self.current = 0.0
        self.voltage = kwargs.get("V0", 0.0)
    def step(self, state, dt):
        self.current += dt
    def get_current(self):
        return self.current
    def get_voltage(self):
        return self.voltage

class DummyCollision(ABC):
    def __init__(self, field_manager=None, **kwargs):
        self.apply_calls = 0
        self.checkpoint_calls = 0
    def apply(self, state, dt):
        self.apply_calls += 1
    def checkpoint(self):
        self.checkpoint_calls += 1
        return {"calls": self.apply_calls}

class DummyDiagnostics:
    def __init__(self, *args, **kwargs):
        self.records = []
        self.checkpoints = []
    def record(self, t, *args, **kwargs):
        self.records.append(t)


def test_run_calls_modules(monkeypatch):
    import sys, types

    solver = DummySolver()

    # Stub external dependencies before importing simulation module
    dummy_trace = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "opencensus", types.SimpleNamespace(trace=dummy_trace))
    monkeypatch.setitem(sys.modules, "opencensus.trace", dummy_trace)

    config_schema_mod = ModuleType("config_schema")
    config_schema_mod.SimulationConfig = type("SimulationConfig", (), {})
    config_schema_mod.FieldManagerConfig = type("FieldManagerConfig", (), {})
    monkeypatch.setitem(sys.modules, "config_schema", config_schema_mod)

    module_registry_mod = ModuleType("module_registry")
    class ModuleRegistry:
        def register(self, *args, **kwargs):
            pass
        def create(self, cls, config=None, field_manager=None, created_modules=None):
            config = config or {}
            return cls(field_manager=field_manager, **config)
    module_registry_mod.ModuleRegistry = ModuleRegistry
    monkeypatch.setitem(sys.modules, "module_registry", module_registry_mod)

    collision_mod = ModuleType("collision_model")
    class CollisionModel(DummyCollision):
        pass
    collision_mod.CollisionModel = CollisionModel
    monkeypatch.setitem(sys.modules, "collision_model", collision_mod)

    radiation_mod = ModuleType("radiation_model")
    class RadiationModel:
        def apply(self, state, dt):
            pass
        def checkpoint(self):
            return {}
    radiation_mod.RadiationModel = RadiationModel
    monkeypatch.setitem(sys.modules, "radiation_model", radiation_mod)

    hybrid_mod = ModuleType("hybrid_controller")
    class HybridController:
        def apply(self, state, dt):
            pass
    hybrid_mod.HybridController = HybridController
    monkeypatch.setitem(sys.modules, "hybrid_controller", hybrid_mod)

    eos_selector_mod = ModuleType("eos_selector")
    eos_selector_mod.select_eos = lambda *a, **k: DummyEOS()
    monkeypatch.setitem(sys.modules, "eos_selector", eos_selector_mod)

    solver_selector_mod = ModuleType("solver_selector")
    solver_selector_mod.select_solver = lambda backend, config, field_manager: solver
    monkeypatch.setitem(sys.modules, "solver_selector", solver_selector_mod)

    circuit_mod = ModuleType("circuit")
    class CircuitModel(DummyCircuit):
        pass
    circuit_mod.CircuitModel = CircuitModel
    monkeypatch.setitem(sys.modules, "circuit", circuit_mod)

    utils_mod = ModuleType("utils")
    class FieldManager:
        def __init__(self, *a, **k):
            pass
        def get_J(self):
            return 0
    class SimulationState:
        def __init__(self, *a, **k):
            pass
    utils_mod.FieldManager = FieldManager
    utils_mod.SimulationState = SimulationState
    monkeypatch.setitem(sys.modules, "utils", utils_mod)

    diagnostics_mod = ModuleType("diagnostics")
    class Diagnostics(DummyDiagnostics):
        pass
    diagnostics_mod.Diagnostics = Diagnostics
    monkeypatch.setitem(sys.modules, "diagnostics", diagnostics_mod)

    pic_solver_mod = ModuleType("pic_solver")
    class PICSolver:
        def __init__(self, *a, **k):
            pass
        def step(self):
            pass
    pic_solver_mod.PICSolver = PICSolver
    monkeypatch.setitem(sys.modules, "pic_solver", pic_solver_mod)

    import dpf2.simulation.dpf_simulation as simmod

    circuit_cfg = SimpleNamespace(C=1.0, V0=1.0, L0=1.0, R0=0.0,
                                 anode_radius=0.1, cathode_radius=0.2,
                                 ESR=0.0, ESL=0.0,
                                 switch_resistance=0.0, switch_inductance=0.0,
                                 transmission_line_impedance=50.0,
                                 transmission_line_length=1.0)
    circuit_cfg.dict = lambda: vars(circuit_cfg)

    collision_cfg = SimpleNamespace()
    collision_cfg.dict = lambda: {}

    diag_cfg = SimpleNamespace(hdf5_filename="diag.h5")
    field_cfg = SimpleNamespace(boundary_conditions={})

    cfg = SimpleNamespace(
        grid_shape=[2, 2, 2],
        dx=1.0,
        dy=1.0,
        dz=1.0,
        domain_lo=(0.0, 0.0, 0.0),
        sim_time=1.0,
        dt_init=None,
        solver_backend="dummy",
        eos_backend="tabulated",
        table_file="table.h5",
        enable_eos_mixture=False,
        mixture_fractions=None,
        circuit=circuit_cfg,
        collision=collision_cfg,
        radiation=None,
        pic=None,
        hybrid=None,
        diagnostics=diag_cfg,
        field_manager=field_cfg,
    )

    sim = simmod.DPFSimulation(cfg)
    sim.run()

    coll = sim.modules["collision"]
    diag = sim.modules["diagnostics"]
    assert sim.current_time == pytest.approx(1.0)
    assert solver.dt_calls == sim.step_count
    assert coll.apply_calls == sim.step_count
    assert coll.checkpoint_calls == sim.step_count
    assert len(diag.records) == sim.step_count
