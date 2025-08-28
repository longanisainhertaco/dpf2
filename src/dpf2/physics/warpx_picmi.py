from __future__ import annotations

"""WarpX PICMI-based PIC driver.

This module provides a small adapter that exposes the minimal :class:`PicDriver`
interface used by the hybrid pinch model.  It relies on the `pywarpx` PICMI API
and optionally the :class:`~dpf2.simulation.warp_piclibrary.PICCollisionHandler`
for Monte Carlo collisions.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np

from .pic_driver import PicDriver
try:  # optional dependency for collisions
    from ..simulation.warp_piclibrary import PICCollisionHandler
except Exception:  # pragma: no cover - collision support optional
    PICCollisionHandler = None  # type: ignore


@dataclass
class WarpXPicmiDriver(PicDriver):
    """Light‑weight PIC driver using the WarpX PICMI interface.

    Parameters
    ----------
    warp:
        Instance of ``picmi.WarpX`` or a compatible object.  Only the very
        small subset of the API exercised in the unit tests is required:
        ``step``, ``get_field`` and ``get_particle_container``.
    collision_handler:
        Optional collision handler implementing ``apply_collisions``.  When
        provided it will be invoked every step with the WarpX instance and the
        timestep.
    """

    warp: object
    collision_handler: Optional[object] = None

    # ------------------------------------------------------------------
    def push_particles(self, dt: float) -> None:
        """Advance particles and fields by ``dt`` using WarpX."""
        # WarpX internally stores the time step so we simply request a single
        # iteration.  For mocks used in the tests we allow an ``advance_particles``
        # method that accepts the time step explicitly.
        if hasattr(self.warp, "advance_particles"):
            self.warp.advance_particles(dt)
        else:  # pragma: no cover - exercised with real WarpX
            self.warp.step(1)

    # ------------------------------------------------------------------
    def exchange_fields(self) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                       Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return the electric and magnetic field components from WarpX."""
        E = tuple(np.array(self.warp.get_field(comp)) for comp in ("Ex", "Ey", "Ez"))
        B = tuple(np.array(self.warp.get_field(comp)) for comp in ("Bx", "By", "Bz"))
        return E, B

    # ------------------------------------------------------------------
    def apply_collisions(self, dt: float) -> None:
        """Apply Monte Carlo collisions using the provided handler."""
        if self.collision_handler is not None:
            self.collision_handler.apply_collisions(self.warp, dt)

    # ------------------------------------------------------------------
    def _diagnostics(self) -> Tuple[float, float]:
        """Compute characteristic radius and total kinetic energy."""
        # We assume a species named ``ions``.  If it does not exist we fall
        # back to the first available particle container.
        container = None
        try:
            container = self.warp.get_particle_container("ions")
        except Exception:  # pragma: no cover - fallback path
            try:
                names = getattr(self.warp, "particle_names", [])
                if names:
                    container = self.warp.get_particle_container(names[0])
            except Exception:
                container = None
        if container is None:
            return 0.0, 0.0
        pos = np.array(container.get_positions())
        vel = np.array(container.get_velocities())
        radius = 0.0
        if len(pos):
            radius = float(sum(math.sqrt(p[0] * p[0] + p[1] * p[1]) for p in pos) / len(pos))
        mass = getattr(container, "mass", 1.0)
        kinetic = 0.0
        for v in vel:
            kinetic += sum(comp * comp for comp in v)
        energy = float(0.5 * mass * kinetic)
        return radius, energy

    # ------------------------------------------------------------------
    def step(self, current: float, dt: float) -> Tuple[float, float]:
        """Advance the PIC model and return ``(radius, energy)``."""
        # The current is not used directly by this simple adapter but is
        # accepted to satisfy the :class:`PicDriver` protocol.
        self.push_particles(dt)
        self.apply_collisions(dt)
        self.exchange_fields()  # pragma: no cover - fields unused in tests
        return self._diagnostics()


__all__ = ["WarpXPicmiDriver"]
