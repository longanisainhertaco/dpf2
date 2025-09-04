import numpy as np

from dpf2.core.bases import CouplingState, PlasmaSolverBase
from dpf2.simulation.hybrid_pic_solver import HybridPICSolver


class FluidStub(PlasmaSolverBase):
    """Minimal fluid solver used for testing."""

    def step(self, state, dt, current, voltage):  # pragma: no cover - trivial
        return state


class ParticleStub:
    def __init__(self, energies):
        # ``numpy_stub.Array`` does not implement rich comparison so we store
        # the energies as a simple list and perform comparisons manually.
        self.energies = list(energies)

    def step(self, state, dt, current, voltage):  # pragma: no cover - trivial
        return state

    def beam_current(self):
        # Count particles exceeding 1 keV and scale by an arbitrary factor
        return float(sum(1 for e in self.energies if e > 1.0))


def test_hybrid_solver_coupling_and_diagnostics():
    fluid = FluidStub()
    particles = ParticleStub([0.5, 2.0, 3.0])
    solver = HybridPICSolver(fluid, particles, dim=2, radius=0.01)

    state = {}
    solver.step(state, dt=1e-9, current=10.0, voltage=5.0)
    feedback = solver.coupling_interface()

    assert isinstance(feedback, CouplingState)
    assert feedback.current == 10.0
    assert solver.last_voltage_spike >= 0.0
    assert solver.last_beam_current == 2.0


