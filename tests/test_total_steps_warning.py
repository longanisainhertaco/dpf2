import types
import sys
import importlib
import importlib.util
from pathlib import Path
import logging

import pytest


def _load_sim_module(monkeypatch):
    """Import ``dpf_simulator_full_backend`` with minimal stubs."""

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

    pydantic_stub = types.ModuleType("pydantic")
    pd_dataclasses = types.ModuleType("pydantic.dataclasses")
    import dataclasses
    pd_dataclasses.dataclass = dataclasses.dataclass
    pydantic_stub.dataclasses = pd_dataclasses
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_stub)
    monkeypatch.setitem(sys.modules, "pydantic.dataclasses", pd_dataclasses)

    # Stub internal modules referenced during import
    package_stub = types.ModuleType("dpf2")
    package_stub.__path__ = []
    simulation_pkg = types.ModuleType("dpf2.simulation")
    simulation_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "dpf2", package_stub)
    monkeypatch.setitem(sys.modules, "dpf2.simulation", simulation_pkg)
    setattr(package_stub, "simulation", simulation_pkg)

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
        "exceptions": ["SimulationRuntimeError"],
    }
    for name, attrs in modules.items():
        mod = types.ModuleType(name)
        for attr in attrs:
            if name == "exceptions" and attr == "SimulationRuntimeError":
                setattr(mod, attr, type(attr, (Exception,), {}))
            else:
                setattr(mod, attr, type(attr, (), {}))
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setitem(sys.modules, f"dpf2.{name}", mod)
        setattr(package_stub, name, mod)

    # Minimal config schema stubs
    config_mod = types.ModuleType("config_schema")
    class SimulationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class SheathConfig:
        """Minimal sheath configuration for test."""
    config_mod.SimulationConfig = SimulationConfig
    config_mod.SheathConfig = SheathConfig
    monkeypatch.setitem(sys.modules, "config_schema", config_mod)

    # Import module under test without package dependencies
    module_path = Path(__file__).resolve().parent.parent / "src/dpf2/simulation/dpf_simulator_full_backend.py"
    spec = importlib.util.spec_from_file_location(
        "dpf2.simulation.dpf_simulator_full_backend", module_path
    )
    sim_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim_mod)
    return sim_mod


def test_warning_on_invalid_dt(monkeypatch, caplog):
    """Ensure a warning is logged and an error raised for invalid dt."""

    sim_mod = _load_sim_module(monkeypatch)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(sim_mod.SimulationRuntimeError):
            sim_mod._estimate_total_steps(1.0, 0.0)

    assert "Invalid dt=0.0: unable to estimate total steps" in caplog.text


def test_failure_logging_on_non_numeric_dt(monkeypatch, caplog):
    """Ensure failures during estimation are logged and raise an error."""

    sim_mod = _load_sim_module(monkeypatch)

    class BadNumber:
        def __float__(self):
            raise TypeError("no float conversion")
        def __le__(self, other):
            return False

    with caplog.at_level(logging.ERROR):
        with pytest.raises(sim_mod.SimulationRuntimeError):
            sim_mod._estimate_total_steps(1.0, BadNumber())

    assert "Failed to estimate total steps" in caplog.text
