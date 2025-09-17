import sys
import types

from dpf2.dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine
from dpf2.core.bases import PlasmaSolverBase, CouplingState
from dpf2.circuit.distributed import TransmissionLineSegment
from dpf2.rlc_solver import solve_distributed_circuit


pic_stub = types.ModuleType("dpf2.simulation.pic_solver")


class PICSolver:  # pragma: no cover - simple stub
    pass


pic_stub.PICSolver = PICSolver
sys.modules.setdefault("dpf2.simulation.pic_solver", pic_stub)


class RecordingCircuit:
    """Minimal circuit solver capturing plasma inductance values."""

    def __init__(self):
        self.currents = [0.0]
        self.voltages = [0.0]
        self.time = [0.0]

        class _Circuit:
            L = 1e-6
            C = 1e-6

        self.circuit = _Circuit()
        self.received_Lp: list[float] = []

    def step(
        self, coupling: CouplingState, back_emf: float, dt: float, energy_tracker=None
    ):
        self.received_Lp.append(coupling.Lp)
        self.currents.append(coupling.current)
        self.voltages.append(coupling.voltage)
        self.time.append(self.time[-1] + dt)
        return CouplingState(current=coupling.current, voltage=coupling.voltage)


class DummyMHD(PlasmaSolverBase):
    """Plasma solver exposing a plasma_inductance method."""

    def step(self, state, dt, current, voltage):
        return (state or 0.0) + 1.0

    def plasma_inductance(self, state):  # pragma: no cover - simple
        return state * 1e-6

    def coupling_interface(self) -> CouplingState:  # pragma: no cover - simple
        return CouplingState()


def test_engine_computes_plasma_inductance(monkeypatch):
    cfg = DPFConfig.with_defaults()
    cfg.simulation_control.time_end = 2e-9
    cfg.simulation_control.min_dt = 1e-9

    engine = SimulationEngine(cfg)
    circuit = RecordingCircuit()
    monkeypatch.setattr(engine, "_setup_circuit", lambda: circuit)
    from dpf2 import pinch_models

    def fake_run(self, t, current):
        zero = t * 0
        return types.SimpleNamespace(
            time=t,
            radius=zero,
            temperature=zero,
            pressure=zero,
            neutron_yield=0.0,
            axial_position=None,
        )

    monkeypatch.setattr(pinch_models.AnalyticPinchModel, "run", fake_run)

    engine.run(plasma_solver=DummyMHD())

    assert circuit.received_Lp and all(lp > 0 for lp in circuit.received_Lp)


class ConstantLpSolver(PlasmaSolverBase):
    """Return a fixed plasma inductance through coupling_interface."""

    def step(self, state, dt, current, voltage):  # pragma: no cover - trivial
        return state

    def coupling_interface(self) -> CouplingState:  # pragma: no cover - trivial
        return CouplingState(Lp=1e-6)


def test_distributed_solver_uses_plasma_inductance():
    seg_L = TransmissionLineSegment(
        0, 1, length=1.0, L_per_m=1e-6, R_per_m=0.0, C_per_m=0.0
    )
    seg_C = TransmissionLineSegment(
        1, 2, length=1.0, L_per_m=0.0, R_per_m=0.0, C_per_m=1e-6
    )
    segments = [seg_L, seg_C]

    res_base = solve_distributed_circuit(segments, [], V0=1.0, t_end=1e-6, dt=1e-7)
    res_lp = solve_distributed_circuit(
        segments, [], V0=1.0, t_end=1e-6, dt=1e-7, em_solver=ConstantLpSolver()
    )

    assert res_lp.current[-1] > res_base.current[-1]
