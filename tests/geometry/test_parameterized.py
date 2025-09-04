import math

from dpf2.geometry import TaperedGeometry, HollowGeometry, ReentrantGeometry


def test_tapered_profile():
    geom = TaperedGeometry(length=1.0, r_base=0.1, r_top=0.05)
    prof = geom.radius_profile(3)
    assert prof[0] == (0.0, 0.1)
    assert prof[-1] == (1.0, 0.05)


def test_hollow_volume():
    geom = HollowGeometry(length=1.0, r_outer=0.1, r_inner=0.05)
    expected = math.pi * (0.1**2 - 0.05**2)
    assert math.isclose(geom.volume(), expected)


def test_reentrant_profile():
    points = [(0.0, 0.1), (0.5, 0.05)]
    geom = ReentrantGeometry(points)
    assert geom.profile() == points
