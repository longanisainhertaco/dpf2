from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple


class PicDriver(Protocol):
    """Minimal interface for an external PIC driver.

    The driver advances kinetic particles and returns diagnostics needed by
    the fluid-hybrid pinch model.  Only the methods required by the tests are
    specified here which keeps the dependency surface extremely small.
    """

    def step(self, current: float, dt: float) -> Tuple[float, float]:
        """Advance the PIC model.

        Parameters
        ----------
        current:
            Circuit current in amperes.
        dt:
            Time step in seconds.

        Returns
        -------
        tuple of float
            A pair ``(radius, energy)`` giving the characteristic plasma
            radius [m] and the particle energy [J] after the step.
        """
        ...


@dataclass
class SimplePicDriver:
    """Very small stand‑in PIC driver used in tests and examples.

    The model evolves a single characteristic radius which shrinks in
    proportion to the supplied circuit current.  The kinetic energy is
    increased by ``current**2`` scaled by ``energy_coeff``.  Both the
    contraction and energy coefficients are chosen simply to give numbers of
    order unity for the unit tests and have no physical significance.
    """

    radius: float = 1e-2
    energy: float = 0.0
    contraction: float = 1e-8
    energy_coeff: float = 1e-6

    def step(self, current: float, dt: float) -> Tuple[float, float]:
        self.radius = max(1e-3, self.radius - current * dt * self.contraction)
        self.energy += current * current * dt * self.energy_coeff
        return self.radius, self.energy


__all__ = ["PicDriver", "SimplePicDriver"]
