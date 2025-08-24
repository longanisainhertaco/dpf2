from math import isclose
from pathlib import Path
import importlib.util
import importlib
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

solver_spec = importlib.util.spec_from_file_location(
    "dpf2.circuit_solver", module_base / "circuit_solver.py"
)
solver_mod = importlib.util.module_from_spec(solver_spec)
sys.modules["dpf2.circuit_solver"] = solver_mod
solver_spec.loader.exec_module(solver_mod)  # type: ignore[misc]
run_circuit_simulation = solver_mod.run_circuit_simulation
CircuitConfig = importlib.import_module("dpf2.circuit_config").CircuitConfig

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
        self.inductance = 0.0
        self.back_emf = 0.0

    def step(self, state, dt, current, voltage):
        super().step(state, dt, current, voltage)
        self.inductance = self.circuit_feedback["Lp"]
        self.back_emf = self.circuit_feedback["emf"]
        return state


def test_energy_conservation():
    L_ext = 1.0
    C_ext = 1.0
    V0 = 1.0
    k = 0.1  # dLp/dt

    cfg = CircuitConfig(
        L_ext=L_ext,
        R_ext=0.0,
        C_ext=C_ext,
        V0=V0,
        switch_delay=0.0,
        switching_model="ideal",
        trigger_jitter_stddev=0.0,
        enable_field_triggered_switch_closure=False,
        abort_on_no_current=False,
    )

    plasma = DummyPlasma(k)

    t, current, voltage, _, _ = run_circuit_simulation(
        cfg, t_end=1e6, num_points=1001, plasma_solver=plasma
    )

    initial_energy = 0.5 * C_ext * V0 ** 2
    final_current = current[-1]
    final_voltage = voltage[-1]
    Lp = plasma.inductance
    final_energy = (
        0.5 * L_ext * final_current ** 2
        + 0.5 * C_ext * final_voltage ** 2
        + 0.5 * Lp * final_current ** 2
    )
    assert isclose(initial_energy, final_energy, rel_tol=1e-3)
