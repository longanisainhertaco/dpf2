from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from ..hall_mhd_solver import MHDState
    from .pic import SimplePIC


class PicDriver(Protocol):
    """Minimal interface for an external PIC driver."""

    def step(self, state: "MHDState", current: float, dt: float) -> Tuple[float, float, float]:
        """Advance the PIC model."""

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
class PhysicalPICDriver:
    """Lightweight physical PIC backend used in tests.

    The driver wraps :class:`dpf2.physics.pic.SimplePIC` to provide a minimal
    particle-in-cell coupling.  Fields and particle distributions are exchanged
    with the fluid model so that unit tests can verify kinetic–fluid
    interaction on three-dimensional grids.
    """

    pic: "SimplePIC"
    field_coeff: float = 1.0
    B_coeff: float = 1.0
    last_E: np.ndarray | None = None
    last_B: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.last_E = np.zeros((1, 1, 1, 3))
        self.last_B = np.zeros((1, 1, 1, 3))

    def step(self, state: "MHDState", current: float, dt: float) -> Tuple[float, float, float]:
        voltage = current * self.field_coeff
        self.pic.step(state, dt, current, voltage)
        pos = np.asarray(self.pic.positions)
        vel = np.asarray(self.pic.velocities)
        radius = float(np.sqrt(np.mean(pos**2))) if pos.size else 0.0
        energy = float(0.5 * self.pic.mass * np.sum(vel**2))
        Ez = voltage / self.pic.length if self.pic.length else 0.0
        By = self.B_coeff * current / (2 * np.pi * max(radius, 1e-6))
        self.last_E[0, 0, 0] = [0.0, 0.0, Ez]
        self.last_B[0, 0, 0] = [0.0, By, 0.0]
        return radius, energy, current

    def exchange_fields(
        self,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        E = (self.last_E[..., 0], self.last_E[..., 1], self.last_E[..., 2])
        B = (self.last_B[..., 0], self.last_B[..., 1], self.last_B[..., 2])
        return E, B

    def exchange_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.zeros((len(self.pic.positions), 3))
        velocities = np.zeros((len(self.pic.velocities), 3))
        if positions.size:
            positions[:, 2] = np.asarray(self.pic.positions)
            velocities[:, 2] = np.asarray(self.pic.velocities)
        return positions, velocities


# ---------------------------------------------------------------------------
# Optional WarpX-based implementation
try:  # pragma: no cover - exercised when WarpX dependency is present
    from .warpx_picmi import WarpXPicmiDriver as _WarpXPicmiDriver
    import numpy as _np

    @dataclass
    class WarpXPICDriver(_WarpXPicmiDriver):  # type: ignore[misc]
        """PIC driver that delegates to the WarpX PICMI interface."""

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


__all__ = ["PicDriver", "PhysicalPICDriver", "WarpXPICDriver"]
