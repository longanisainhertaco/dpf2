import numpy as np
import pytest

from dpf2.physics.gv_front import GVFront


def test_gv_front_shape():
    gv = GVFront(anode_radius=0.05, velocity=2e4)
    z = np.linspace(0, gv.anode_radius, 5)
    r = gv.radius(z)
    expected = np.array([gv.anode_radius**2] * len(z))
    assert np.allclose(r**2 + z**2, expected)


def test_gv_front_arrival_time():
    gv = GVFront(anode_radius=0.05, velocity=1e5)
    z = 0.02
    assert gv.arrival_time(z) == pytest.approx(z / gv.velocity)
