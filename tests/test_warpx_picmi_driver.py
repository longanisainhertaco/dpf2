import numpy as np

from dpf2.physics.warpx_picmi import WarpXPicmiDriver


class MockParticleContainer:
    def __init__(self):
        self.mass = 2.0
        self._pos = [[0.0, 0.0, 0.0]]
        self._vel = [[1.0, 0.0, 0.0]]

    def get_positions(self):
        return [list(p) for p in self._pos]

    def set_positions(self, pts):
        self._pos = [list(p) for p in pts]

    def get_velocities(self):
        return [list(v) for v in self._vel]

    def set_velocities(self, v):
        self._vel = [list(vv) for vv in v]


class MockWarpX:
    def __init__(self):
        self.pc = MockParticleContainer()
        self.dt = 1.0

    def get_particle_container(self, name):
        return self.pc

    def advance_particles(self, dt):
        # Simple push: increase velocity in x and move particle
        v = np.array(self.pc.get_velocities())
        v[:, 0] += 1.0
        self.pc.set_velocities(v)
        x = np.array(self.pc.get_positions()) + v * dt
        self.pc.set_positions(x)

    def get_field(self, comp):
        return [0.0]


def test_driver_updates_velocity_and_energy():
    warp = MockWarpX()
    driver = WarpXPicmiDriver(warp)
    r, e = driver.step(current=0.0, dt=warp.dt)
    # Velocity increased by 1.0 in advance_particles
    assert abs(warp.pc.get_velocities()[0][0] - 2.0) < 1e-12
    # Position updated accordingly giving radius 2.0
    assert abs(r - 2.0) < 1e-12
    # Energy = 0.5 * m * v^2 with m=2 and v=2
    assert abs(e - 0.5 * warp.pc.mass * 4.0) < 1e-12
