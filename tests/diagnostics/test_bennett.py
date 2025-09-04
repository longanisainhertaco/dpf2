import numpy as np

from dpf2.diagnostics import bennett_radius


def test_bennett_radius_monotonic():
    n = 1e20
    T = 1e3
    currents = np.linspace(1e3, 1e5, 10)
    radii = [bennett_radius(I, n, T) for I in currents]
    assert all(r1 >= r2 for r1, r2 in zip(radii[:-1], radii[1:]))
