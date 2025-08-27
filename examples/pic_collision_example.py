"""Minimal example demonstrating the :class:`PICCollisionHandler` API.

The real WarpX runtime is not required for this example.  Instead a very small
mock of the WarpX particle container interface is used so the example can be
run as a standalone script::

    $ python examples/pic_collision_example.py

The script prints the electron velocities before and after a Monte Carlo
collision step with ions.
"""

from __future__ import annotations

import random

from dpf2.simulation.warp_piclibrary import PICCollisionHandler


class SimpleParticleContainer:
    """Light‑weight stand in for a WarpX particle container."""

    def __init__(self, velocities, mass=1.0):
        self._vel = [list(v) for v in velocities]
        self.mass = mass

    def get_velocities(self):
        return [v[:] for v in self._vel]

    def set_velocities(self, v):
        self._vel = [list(val) for val in v]


class SimpleWarpX:
    """Minimal subset of the WarpX API required by :class:`PICCollisionHandler`."""

    def __init__(self):
        self.volume = 1.0
        self._species = {
            "e": SimpleParticleContainer([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], mass=1.0),
            "i": SimpleParticleContainer([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], mass=1836.0),
        }

    def get_particle_container(self, name):
        return self._species[name]


def constant_nu(ne, Te, **kwargs):
    """Return a constant collision frequency for demonstration."""

    return 100.0


def main() -> None:
    warp = SimpleWarpX()
    handler = PICCollisionHandler(constant_nu, species_pairs=[("e", "i")])
    dt = 0.1

    print("electron velocities before:")
    print(warp.get_particle_container("e").get_velocities())

    handler.apply_collisions(warp, dt)

    print("electron velocities after:")
    print(warp.get_particle_container("e").get_velocities())


if __name__ == "__main__":  # pragma: no cover - example script
    random.seed(1)
    main()
