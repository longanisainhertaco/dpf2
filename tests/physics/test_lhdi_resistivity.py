import numpy as np
import pytest

from dpf2.physics import compute_effective_eta


def test_compute_effective_eta_basic():
    eta, e = compute_effective_eta(4.0, 2.0)
    assert float(eta) == pytest.approx(1.0)
    assert float(e) == pytest.approx(2.0)


def test_compute_effective_eta_array_broadcast():
    power = np.array([2.0, 8.0])
    phase = np.array([1.0, 2.0])
    eta, e = compute_effective_eta(power, phase)
    assert np.allclose(eta, [2.0, 2.0])
    assert np.allclose(e, [2.0, 4.0])
