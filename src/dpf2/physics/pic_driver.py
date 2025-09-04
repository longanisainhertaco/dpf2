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
        # need to ensure particles are exchanged each time it is invoked.
        def step(self, current: float, dt: float) -> Tuple[float, float]:
            radius, energy = super().step(current, dt)
            self.exchange_particles()
            return radius, energy

except Exception:  # pragma: no cover - fallback when WarpX is unavailable
    class WarpXPICDriver(PicDriver):  # type: ignore[misc]
        """Fallback stub used when WarpX is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("WarpX PICMI interface is not available")


__all__ = ["PicDriver", "SimplePicDriver", "WarpXPICDriver"]
