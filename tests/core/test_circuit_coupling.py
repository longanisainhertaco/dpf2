from math import isclose
from pathlib import Path
import importlib.util
import sys
import types

module_base = Path(__file__).resolve().parents[2] / "src/dpf2"

pkg = types.ModuleType("dpf2")
pkg.__path__ = [str(module_base)]  # type: ignore[attr-defined]
core_pkg = types.ModuleType("dpf2.core")
core_pkg.__path__ = [str(module_base / "core")]  # type: ignore[attr-defined]
physics_pkg = types.ModuleType("dpf2.physics")
physics_pkg.__path__ = [str(module_base / "physics")]  # type: ignore[attr-defined]
sys.modules.setdefault("dpf2", pkg)
sys.modules.setdefault("dpf2.core", core_pkg)
sys.modules.setdefault("dpf2.physics", physics_pkg)

circuit_spec = importlib.util.spec_from_file_location(
    "dpf2.core.circuit", module_base / "core/circuit.py"
)
circuit_mod = importlib.util.module_from_spec(circuit_spec)
sys.modules["dpf2.core.circuit"] = circuit_mod
circuit_spec.loader.exec_module(circuit_mod)  # type: ignore[misc]
RLCCircuitSolver = circuit_mod.RLCCircuitSolver

plasma_spec = importlib.util.spec_from_file_location(
    "dpf2.physics.simple_plasma", module_base / "physics/simple_plasma.py"
)
plasma_mod = importlib.util.module_from_spec(plasma_spec)
sys.modules["dpf2.physics.simple_plasma"] = plasma_mod
plasma_spec.loader.exec_module(plasma_mod)  # type: ignore[misc]
ZeroDPlasma = plasma_mod.ZeroDPlasma


class DummyPlasma(ZeroDPlasma):
    """Plasma with linearly increasing inductance."""

    def __init__(self, k: float):
        def model(t, current, voltage):
            Lp = k * t
            emf = k * current
            return Lp, emf
        super().__init__(model)


def test_energy_conservation():
    L_ext = 1.0
    C_ext = 1.0
    V0 = 1.0
    k = 0.1  # dLp/dt

    circuit = RLCCircuitSolver(L_ext=L_ext, R_ext=0.0, C_ext=C_ext, V0=V0)
    plasma = DummyPlasma(k)

    dt = 1e-3
    steps = 1000

    current = circuit.currents[-1]
    voltage = circuit.voltages[-1]
    plasma.step(None, 0.0, current, voltage)
    for _ in range(steps):
        current, voltage = circuit.step(current, 0.0, dt, plasma.circuit_feedback)
        plasma.step(None, dt, current, voltage)

    initial_energy = 0.5 * C_ext * V0 ** 2
    Lp = plasma.circuit_feedback["Lp"]
    final_energy = 0.5 * L_ext * current ** 2 + 0.5 * C_ext * voltage ** 2 + 0.5 * Lp * current ** 2
    assert isclose(initial_energy, final_energy, rel_tol=1e-3)
