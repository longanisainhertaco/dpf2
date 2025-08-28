import math

from dpf2.core.circuit import RLCCircuitSolver
from dpf2.core.bases import PlasmaSolverBase, CouplingState
from dpf2.geometry.inductance import coaxial_inductance


class AnalyticPlasma(PlasmaSolverBase):
    def __init__(self, r0=0.01, drdt=-1.0, outer=0.02, length=0.1):
        self.time = 0.0
        self.radius = r0
        self.r0 = r0
        self.drdt = drdt
        self.outer = outer
        self.length = length
        self.history: list[tuple[float, float]] = []

    def step(self, state, dt, current, voltage):
        self.time += dt
        self.radius = self.r0 + self.drdt * self.time
        return state

    def coupling_interface(self):
        Lp = coaxial_inductance(self.radius, self.outer, self.length)
        self.history.append((self.time, Lp))
        return CouplingState(Lp=Lp, emf=0.0)


def test_inductance_profile_recovery():
    plasma = AnalyticPlasma()
    circuit = RLCCircuitSolver(L_ext=1.0, R_ext=0.0, C_ext=1.0, V0=1.0)
    steps = 20
    dt = 1e-6 / (steps - 1)

    current = circuit.currents[-1]
    voltage = circuit.voltages[-1]
    plasma.step(None, 0.0, current, voltage)
    for _ in range(1, steps):
        feedback = plasma.coupling_interface()
        feedback.current = current
        feedback.voltage = voltage
        updated = circuit.step(feedback, 0.0, dt)
        plasma.step(None, dt, updated.current, updated.voltage)
        current, voltage = updated.current, updated.voltage

    times = [t for t, _ in plasma.history]
    Lp_vals = [Lp for _, Lp in plasma.history]
    expected = [
        coaxial_inductance(plasma.r0 + plasma.drdt * t, plasma.outer, plasma.length)
        for t in times
    ]
    for a, b in zip(Lp_vals, expected):
        assert math.isclose(a, b, rel_tol=1e-6)
