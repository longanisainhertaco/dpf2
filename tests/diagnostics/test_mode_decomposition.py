import math
import numpy as np

from dpf2.diagnostics.modes import azimuthal_mode_spectrum, growth_rate


def test_mode_spectrum_simple_cosine():
    theta = np.linspace(0.0, 2 * np.pi, 17)[:-1]
    r = np.linspace(0.0, 1.0, 4)
    field = np.zeros((len(r), len(theta)))
    # ``numpy_stub`` lacks ``cos`` – emulate via phase-shifted sine
    cos = lambda x: np.sin(x + np.pi / 2)
    for i in range(len(r)):
        field[i] = 0.5 * cos(theta) + 0.25 * cos(2 * theta)
        field[i] = field[i] + 1.0
    spec = azimuthal_mode_spectrum(field, axis=1)
    assert spec.shape[0] >= 3
    assert np.isclose(spec[0], 1.0, atol=1e-12)
    assert np.isclose(spec[1], 0.5, atol=1e-12)
    assert np.isclose(spec[2], 0.25, atol=1e-12)


def test_growth_rate():
    prev = np.array([1.0, 1.0, 1.0])
    curr = np.array([1.0, math.e, 0.5])
    dt = 1.0
    rates = growth_rate(prev, curr, dt)
    assert np.allclose(rates, [0.0, 1.0, math.log(0.5)])
