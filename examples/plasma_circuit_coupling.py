"""Example of circuit/plasma coupling with energy conservation."""

from pathlib import Path
import importlib.util
import sys
import types

module_base = Path(__file__).resolve().parents[1] / "src/dpf2"

pkg = types.ModuleType("dpf2")
pkg.__path__ = [str(module_base)]  # type: ignore[attr-defined]
core_pkg = types.ModuleType("dpf2.core")
core_pkg.__path__ = [str(module_base / "core")]  # type: ignore[attr-defined]
physics_pkg = types.ModuleType("dpf2.physics")
physics_pkg.__path__ = [str(module_base / "physics")]  # type: ignore[attr-defined]
sys.modules.setdefault("dpf2", pkg)
sys.modules.setdefault("dpf2.core", core_pkg)
sys.modules.setdefault("dpf2.physics", physics_pkg)

core_spec = importlib.util.spec_from_file_location(
    "dpf2.core.circuit", module_base / "core/circuit.py"
)
core_mod = importlib.util.module_from_spec(core_spec)
sys.modules["dpf2.core.circuit"] = core_mod
core_spec.loader.exec_module(core_mod)  # type: ignore[misc]
RLCCircuitSolver = core_mod.RLCCircuitSolver

plasma_spec = importlib.util.spec_from_file_location(
    "dpf2.physics.simple_plasma", module_base / "physics/simple_plasma.py"
)
plasma_mod = importlib.util.module_from_spec(plasma_spec)
sys.modules["dpf2.physics.simple_plasma"] = plasma_mod
plasma_spec.loader.exec_module(plasma_mod)  # type: ignore[misc]
ZeroDPlasma = plasma_mod.ZeroDPlasma


def inductance_model(t: float, current: float, voltage: float) -> tuple[float, float]:
    """Return plasma inductance and back‑EMF.

    The inductance grows linearly with time which induces an opposing EMF
    proportional to the current.
    """
    k = 0.1
    Lp = k * t
    emf = k * current
    return Lp, emf


def main() -> None:
    circuit = RLCCircuitSolver(L_ext=1.0, R_ext=0.0, C_ext=1.0, V0=1.0)
    plasma = ZeroDPlasma(inductance_model)

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
    print(f"Initial energy: {initial_energy:.6f}")
    print(f"Final energy:   {final_energy:.6f}")
    print(f"Difference:      {final_energy - initial_energy:.2e}")


if __name__ == "__main__":
    main()
