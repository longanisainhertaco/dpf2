from math import isclose
from pathlib import Path
import importlib.util
import sys
import types

a = Path(__file__).resolve().parents[2] / "src/dpf2"

pkg = types.ModuleType("dpf2")
pkg.__path__ = [str(a)]  # type: ignore[attr-defined]
core_pkg = types.ModuleType("dpf2.core")
core_pkg.__path__ = [str(a / "core")]  # type: ignore[attr-defined]
physics_pkg = types.ModuleType("dpf2.physics")
physics_pkg.__path__ = [str(a / "physics")]  # type: ignore[attr-defined]
sys.modules.setdefault("dpf2", pkg)
sys.modules.setdefault("dpf2.core", core_pkg)
sys.modules.setdefault("dpf2.physics", physics_pkg)

core_spec = importlib.util.spec_from_file_location(
    "dpf2.core.circuit", a / "core/circuit.py"
)
core_mod = importlib.util.module_from_spec(core_spec)
sys.modules["dpf2.core.circuit"] = core_mod
core_spec.loader.exec_module(core_mod)  # type: ignore[misc]
RLCCircuitSolver = core_mod.RLCCircuitSolver

plasma_spec = importlib.util.spec_from_file_location(
    "dpf2.physics.simple_plasma", a / "physics/simple_plasma.py"
)
plasma_mod = importlib.util.module_from_spec(plasma_spec)
sys.modules["dpf2.physics.simple_plasma"] = plasma_mod
plasma_spec.loader.exec_module(plasma_mod)  # type: ignore[misc]
ZeroDPlasma = plasma_mod.ZeroDPlasma


def test_energy_conservation():
    """Circuit/plasma coupling conserves total energy."""

    circuit = RLCCircuitSolver(L_ext=1.0, R_ext=0.0, C_ext=1.0, V0=1.0)

    k = 0.1

    def model(t, current, voltage):
        Lp = k * t
        emf = k * current
        return Lp, emf

    plasma = ZeroDPlasma(model)

    dt = 1e-3
    steps = 1000

    current = circuit.currents[-1]
    voltage = circuit.voltages[-1]
    plasma.step(None, 0.0, current, voltage)

    for _ in range(steps):
        feedback = {"Lp": plasma.inductance, "emf": plasma.back_emf}
        current, voltage = circuit.step(current, 0.0, dt, feedback)
        plasma.step(None, dt, current, voltage)

    Lp = plasma.inductance
    initial_energy = 0.5 * circuit.C_ext * circuit.V0 ** 2
    final_energy = (
        0.5 * circuit.L_ext * current ** 2
        + 0.5 * circuit.C_ext * voltage ** 2
        + 0.5 * Lp * current ** 2
    )
    assert isclose(initial_energy, final_energy, rel_tol=1e-3)
