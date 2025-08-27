from __future__ import annotations

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
