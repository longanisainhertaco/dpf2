import numpy as np
import pytest

from dpf2.pinch_models import MHDPinchModel, HybridPinchModel


class DummyPIC:
    def __init__(self, radius0: float = 1.0) -> None:
        self.radius = radius0
        self.energy = 0.0

    def step(self, state, current: float, dt: float) -> tuple[float, float, float]:
        # Simple contraction model independent of current for testing
        self.radius *= 1.0 - 0.1 * dt
        return self.radius, self.energy, current


class DummySolver:
    def step(self, state, dt, current=0.0):
        return state


def test_mhd_pinch_energy_conservation_and_radius(monkeypatch):
    monkeypatch.setattr("dpf2.pinch_models.HallMHDSolver", DummySolver)
    model = MHDPinchModel()
    t = np.linspace(0.0, 1e-7, 3)
    I = np.zeros_like(t)
    res = model.run(t, I)
    assert np.isclose(res.energy[0], res.energy[-1])

    model2 = MHDPinchModel()
    t2 = np.linspace(0.0, 1e-6, 5)
    I2 = np.linspace(0.0, 1e5, 5)
    res2 = model2.run(t2, I2)
    assert np.isclose(res2.radius[-1], res2.radius[0])


def test_hybrid_pinch_energy_conservation_and_radius(monkeypatch):
    pic = DummyPIC()
    monkeypatch.setattr("dpf2.pinch_models.HallMHDSolver", DummySolver)
    model = HybridPinchModel(pic_driver=pic)
    t = np.linspace(0.0, 1e-7, 3)
    I = np.zeros_like(t)
    res = model.run(t, I)
    assert np.isclose(res.energy[0], res.energy[-1])

    pic2 = DummyPIC()
    model2 = HybridPinchModel(pic_driver=pic2)
    t2 = np.linspace(0.0, 1e-6, 5)
    I2 = np.linspace(0.0, 1e5, 5)
    res2 = model2.run(t2, I2)
    assert res2.radius[-1] < res2.radius[0]
