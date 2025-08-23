import sys
import types
import importlib.util
from pathlib import Path

import pytest


def _load_simulation(monkeypatch):
    """Load DPFSimulation with minimal dependency stubs."""
    # Stub numpy with basic linspace
    numpy_stub = types.ModuleType("numpy")

    def linspace(start, stop, num):
        if num <= 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]

    numpy_stub.linspace = linspace
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    # Stub h5py to write simple text files
    class _FakeH5:
        class File:
            def __init__(self, fname, mode):
                self.fname = fname
                self.data = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                with open(self.fname, "w") as f:
                    for k, v in self.data.items():
                        f.write(f"{k}:{v}\n")

            def create_dataset(self, key, data):
                self.data[key] = data

    monkeypatch.setitem(sys.modules, "h5py", _FakeH5())

    # Stub config module to avoid pydantic
    config_stub = types.ModuleType("dpf2.core.config")
    class DPFConfig:  # minimal placeholder
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    config_stub.DPFConfig = DPFConfig
    monkeypatch.setitem(sys.modules, "dpf2.core.config", config_stub)

    # Create dummy package structure
    repo_root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("dpf2")
    pkg.__path__ = [str(repo_root / "src" / "dpf2")]
    core_pkg = types.ModuleType("dpf2.core")
    core_pkg.__path__ = [str(repo_root / "src" / "dpf2" / "core")]
    monkeypatch.setitem(sys.modules, "dpf2", pkg)
    monkeypatch.setitem(sys.modules, "dpf2.core", core_pkg)

    spec = importlib.util.spec_from_file_location(
        "dpf2.core.simulation", repo_root / "src" / "dpf2" / "core" / "simulation.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "dpf2.core.simulation", module)
    spec.loader.exec_module(module)
    return module.DPFSimulation


def test_simulation_advances_and_writes(tmp_path, monkeypatch):
    DPFSimulation = _load_simulation(monkeypatch)

    class SimpleConfig:
        anode_radius = 0.025
        electrode_length = 0.10
        nr_cells = 10
        nz_cells = 10
        cfl_number = 1e-3
        end_time = 1e-6
        charging_voltage = 15000.0

    class DummyPlasmaSolver:
        def __init__(self):
            self.calls = 0

        def step(self, state, dt):
            self.calls += 1
            return (state or 0.0) + dt

    class DummyCircuitSolver:
        def step(self, current, voltage, dt):
            return current + dt, voltage - dt

    cfg = SimpleConfig()
    plasma = DummyPlasmaSolver()
    circuit = DummyCircuitSolver()
    sim = DPFSimulation(cfg, plasma_solver=plasma, circuit_solver=circuit)
    sim.run(output_dir=str(tmp_path), output_interval=5e-7)

    # state should evolve
    assert plasma.calls > 0
    assert sim.current > 0.0
    assert sim.voltage < cfg.charging_voltage

    # output files should be written
    files = sorted(tmp_path.glob("data_*.h5"))
    assert len(files) >= 2
