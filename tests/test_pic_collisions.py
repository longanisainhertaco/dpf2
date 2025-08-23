import numpy as np
import pytest

from Simulation.warp_piclibrary import PICCollisionHandler


class SimpleParticleContainer:
    def __init__(self, velocities, mass=1.0):
        self._vel = np.array(velocities, dtype=float)
        self.mass = mass

    def get_velocities(self):
        return self._vel.copy()

    def set_velocities(self, v):
        self._vel = np.array(v, dtype=float)


class SimpleWarpX:
    def __init__(self):
        self._species = {
            "e": SimpleParticleContainer([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], mass=1.0),
            "i": SimpleParticleContainer([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], mass=1.0),
        }
        self.registered_ops = []
        self.volume = 1.0

    def get_particle_container(self, name):
        return self._species[name]

    def add_collision_operator(self, s1, s2, freq_func, kwargs):
        self.registered_ops.append((s1, s2))


def test_apply_collisions_scatter_velocities():
    np.random.seed(1)
    warp = SimpleWarpX()
    handler = PICCollisionHandler(lambda ne, Te, Z=1.0: 100.0)

    v_e_before = warp.get_particle_container("e").get_velocities()
    v_i_before = warp.get_particle_container("i").get_velocities()

    handler.apply_collisions("e", "i", warp, dt=0.1)

    v_e_after = warp.get_particle_container("e").get_velocities()
    v_i_after = warp.get_particle_container("i").get_velocities()

    assert not np.allclose(v_e_before, v_e_after)
    assert not np.allclose(v_i_before, v_i_after)


def test_apply_collisions_unknown_species():
    warp = SimpleWarpX()
    handler = PICCollisionHandler(lambda ne, Te: 1.0)
    with pytest.raises(ValueError):
        handler.apply_collisions("e", "x", warp, dt=0.1)


def test_apply_collisions_missing_hook():
    class BrokenWarp:
        pass

    warp = BrokenWarp()
    handler = PICCollisionHandler(lambda ne, Te: 1.0)
    with pytest.raises(AttributeError):
        handler.apply_collisions("e", "i", warp, dt=0.1)


def test_setup_warpx_collisions_registers_ops():
    warp = SimpleWarpX()
    handler = PICCollisionHandler(lambda ne, Te, Z=1.0: 1.0)
    handler.setup_warpx_collisions(warp, [("e", "i")])
    assert ("e", "i") in warp.registered_ops
