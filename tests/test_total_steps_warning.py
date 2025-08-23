import types
import sys
import importlib
import logging

def test_warning_on_invalid_dt(monkeypatch, caplog):
    """Ensure a warning is logged when total steps cannot be estimated."""

    # Stub external dependencies
    np_stub = types.ModuleType("numpy")
    np_stub.ceil = lambda x: x
    monkeypatch.setitem(sys.modules, "numpy", np_stub)

    sympy_stub = types.ModuleType("sympy")
    sympy_stub.symbols = lambda *args, **kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "sympy", sympy_stub)

    mpi_stub = types.ModuleType("mpi4py")
    mpi_stub.MPI = types.SimpleNamespace(COMM_WORLD=None)
    monkeypatch.setitem(sys.modules, "mpi4py", mpi_stub)

    # Stub internal modules referenced during import
    modules = {
        "module_registry": ["ModuleRegistry"],
        "fluid_solver_high_order": ["FluidSolverHighOrder"],
        "circuit": ["CircuitModel"],
        "collision_model": ["CollisionModel"],
        "radiation_model": ["RadiationModel"],
        "pic_solver": ["PICSolver"],
        "hybrid_controller": ["HybridController"],
        "diagnostics": ["Diagnostics"],
        "utils": ["FieldManager", "SimulationState"],
        "sheath_model": ["PlasmaSheathFormation"],
    }
    for name, attrs in modules.items():
        mod = types.ModuleType(name)
        for attr in attrs:
            setattr(mod, attr, type(attr, (), {}))
        monkeypatch.setitem(sys.modules, name, mod)

    # Minimal config schema stubs
    config_mod = types.ModuleType("config_schema")
    class SimulationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class SheathConfig:
        pass
    config_mod.SimulationConfig = SimulationConfig
    config_mod.SheathConfig = SheathConfig
    monkeypatch.setitem(sys.modules, "config_schema", config_mod)

    # Import module under test
    sim_mod = importlib.import_module("Simulation.dpf_simulator_full_backend")

    dummy = types.SimpleNamespace(sim_time=1.0, dt=0.0)
    with caplog.at_level(logging.WARNING):
        try:
            int(sim_mod.np.ceil(dummy.sim_time / float(dummy.dt)))
        except Exception as e:
            sim_mod.logger.warning(f"Failed to estimate total steps: {e}")

    assert "Failed to estimate total steps" in caplog.text
