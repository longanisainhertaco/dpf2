from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from ..hall_mhd_solver import MHDState


class PicDriver(Protocol):
    """Minimal interface for an external PIC driver.

    The driver advances kinetic particles and exchanges data with the
    fluid-hybrid pinch model.  Only the very small subset of functionality
    exercised in the unit tests is defined here which keeps the dependency
    surface extremely small.
    """

    def step(self, state: "MHDState", current: float, dt: float) -> Tuple[float, float, float]:
        """Advance the PIC model.

        Parameters
        ----------
        state:
            Full MHD state at the beginning of the time step.
        current:
            Circuit current in amperes.
        dt:
            Time step in seconds.

        Returns
        -------
        tuple of float
            A triple ``(radius, energy, current)`` giving the characteristic
            plasma radius [m], the particle energy [J] and the effective
            current [A] to be applied to the fluid region after the step.
        """
        ...

    def exchange_fields(
        self,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Return electric and magnetic field components."""

    def exchange_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return particle positions and velocities."""


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

    def step(self, state: "MHDState", current: float, dt: float) -> Tuple[float, float, float]:
        self.radius = max(1e-3, self.radius - current * dt * self.contraction)
        self.energy += current * current * dt * self.energy_coeff
        return self.radius, self.energy, current

    def exchange_fields(
        self,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        return (np.empty(0), np.empty(0), np.empty(0)), (
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )

    def exchange_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty((0, 3)), np.empty((0, 3))


# ---------------------------------------------------------------------------
# Optional WarpX-based implementation
try:  # pragma: no cover - exercised when WarpX dependency is present
    from .warpx_picmi import WarpXPicmiDriver as _WarpXPicmiDriver
    import numpy as _np

    @dataclass
    class WarpXPICDriver(_WarpXPicmiDriver):  # type: ignore[misc]
        """PIC driver that delegates to the WarpX PICMI interface.

        This thin wrapper re-exposes :class:`WarpXPicmiDriver` under a more
        convenient name and adds a simple particle exchange method used by the
        hybrid pinch model.  The heavy lifting is provided by the
        :mod:`dpf2.physics.warpx_picmi` module which in turn relies on the
        `pywarpx` package.
        """

        def exchange_particles(self) -> Tuple[_np.ndarray, _np.ndarray]:
            """Return particle positions and velocities from WarpX."""
            try:
                container = self.warp.get_particle_container("ions")
            except Exception:  # pragma: no cover - exercised in tests via mocks
                try:
                    names = getattr(self.warp, "particle_names", [])
                    container = (
                        self.warp.get_particle_container(names[0]) if names else None
                    )
                except Exception:
                    container = None
            if container is None:
                return _np.empty((0, 3)), _np.empty((0, 3))
            pos = _np.array(container.get_positions())
            vel = _np.array(container.get_velocities())
            return pos, vel

        # ``step`` from ``WarpXPicmiDriver`` already exchanges fields; we only
        # need to ensure particles are exchanged each time it is invoked and
        # return the effective current for the fluid solver.
        def step(
            self, state: "MHDState", current: float, dt: float
        ) -> Tuple[float, float, float]:
            radius, energy = super().step(current, dt)
            self.exchange_particles()
            return radius, energy, current

except Exception:  # pragma: no cover - fallback when WarpX is unavailable
    class WarpXPICDriver(PicDriver):  # type: ignore[misc]
        """Fallback stub used when WarpX is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("WarpX PICMI interface is not available")


__all__ = ["PicDriver", "SimplePicDriver", "WarpXPICDriver"]
