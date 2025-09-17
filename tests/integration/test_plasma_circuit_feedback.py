import numpy as np
from dpf2.simulation.circuit import CircuitModel


class DummyFieldManager:
    def __init__(self, *args, **kwargs):
        pass

    def get_J(self):
        return 0.0


class DummyCollision:
    def spitzer_resistivity(self, ne, Te, lnL):
        return 0.0


def make_circuit():
    fm = DummyFieldManager()
    return CircuitModel(
        C=1e-6,
        L0=1e-6,
        R0=0.0,
        anode_radius=0.01,
        cathode_radius=0.02,
        collision_model=DummyCollision(),
        field_manager=fm,
        V0=1000.0,
    )


from dpf2.core.bases import CouplingState


def run_trace(inductances):
    circuit = make_circuit()
    dt = 1e-7
    trace = []
    for Lp in inductances:
        current = circuit.get_current()
        voltage = circuit.get_voltage()
        coupling = CouplingState(Lp=Lp, emf=0.0, current=current, voltage=voltage)
        updated = circuit.step(coupling, 0.0, dt)
        trace.append(updated.current)
    return np.array(trace)


def test_current_changes_with_plasma_inductance():
    steps = [0.0] * 5
    base = run_trace(steps)
    varying = run_trace([i * 1e-7 for i in range(5)])
    assert not np.allclose(base, varying)
