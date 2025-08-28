"""Demonstration of hybrid plasma–circuit feedback using :class:`HybridPinchModel`.

The example runs a tiny time stepping loop with a mock PIC driver and the
``CircuitSolver``.  The circuit current drives the plasma which in turn feeds
back an artificial back electromotive force based on the particle energy.
This showcases the bidirectional coupling used in hybrid simulations.
"""

from __future__ import annotations

import numpy as np

from dpf2.pinch_models import HybridPinchModel
from dpf2.physics.pic_driver import PicDriver
from dpf2.circuit_solver import CircuitSolver, RLCCircuit
from dpf2.core.bases import CouplingState


class MockPicDriver(PicDriver):
    """Very small stand‑in for a PIC code used in the example."""

    def __init__(self, radius: float = 1e-2):
        self.radius = radius
        self.energy = 0.0

    def step(self, current: float, dt: float):
        # Collapse radius proportional to current and accumulate kinetic energy
        self.radius = max(1e-3, self.radius - current * dt * 1e-8)
        self.energy += current ** 2 * dt * 1e-6
        return self.radius, self.energy


if __name__ == "__main__":
    pic = MockPicDriver()
    model = HybridPinchModel(pic_driver=pic, switch_radius=5e-3)
    circuit = CircuitSolver(RLCCircuit(L=10e-9, R=0.1, C=1e-6, V0=2000))

    t = np.linspace(0.0, 1e-6, 100)
    current = 0.0
    radius = pic.radius
    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        # Plasma responds to circuit current
        radius, energy = pic.step(current, dt)
        # Simple feedback: plasma energy generates back EMF
        back_emf = energy * 1e-3
        coupling = CouplingState(current=current, voltage=circuit.voltages[-1])
        updated = circuit.step(coupling, back_emf, dt)
        current = updated.current

    print(f"Final current: {current:.3e} A")
    print(f"Final radius:  {radius:.3e} m")
