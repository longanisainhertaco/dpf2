import math
from dpf2.plasma_model import advance_plasma_with_circuit
from dpf2.physics.simple_plasma import ZeroDPlasma
from dpf2.core.bases import CouplingState
from dpf2.geometry.inductance import loop_mutual_inductance


class State:
    def __init__(self, radius: float, axial_position: float = 0.0) -> None:
        self.radius = radius
        self.axial_position = axial_position


def test_geometry_mutual_inductance_updates() -> None:
    # Dummy plasma solver with no intrinsic coupling
    plasma = ZeroDPlasma(lambda t, I, V: (0.0, 0.0))
    plasma.coil_radius = 0.1  # external circuit radius
    state = State(radius=0.05)
    coupling = CouplingState(current=1.0, voltage=0.0, mutual_inductance=0.0)
    dt = 1e-6

    updated = advance_plasma_with_circuit(plasma, state, coupling, dt)
    expected_M = loop_mutual_inductance(0.1, 0.05, 0.0)
    assert math.isclose(updated.mutual_inductance, expected_M)
    assert math.isclose(updated.back_reaction, expected_M / dt)

    # Change radius and ensure back reaction reflects dM/dt
    state.radius = 0.06
    updated2 = advance_plasma_with_circuit(plasma, state, updated, dt)
    expected_M2 = loop_mutual_inductance(0.1, 0.06, 0.0)
    dMdt = (expected_M2 - expected_M) / dt
    assert math.isclose(updated2.back_reaction, updated.current * dMdt)


from dpf2.circuit_solver import CircuitSolver, RLCCircuit


def test_circuit_solver_requests_update():
    circ = CircuitSolver(RLCCircuit(L=1e-6, R=0.0, C=1e-6, V0=0.0))
    coupling = CouplingState(current=0.0, voltage=0.0)

    def updater(I: float, V: float) -> CouplingState:
        return CouplingState(Lp=0.0, emf=0.0, mutual_inductance=0.2, back_reaction=0.3)

    res = circ.step(coupling, 0.0, 1e-6, update_coupling=updater)
    assert math.isclose(res.mutual_inductance, 0.2)
    assert math.isclose(res.back_reaction, 0.3)
