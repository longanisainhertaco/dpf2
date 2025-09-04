import numpy as np
import pytest

from dpf2.pinch_models import HybridPinchModel
from dpf2.physics.pic_driver import PicDriver


class DummyPIC(PicDriver):
    """Simple mock PIC driver returning prescribed radii and energies."""

    def __init__(self):
        self.calls = 0
        self.radii = [0.1, 0.2, 0.3]
        self.energy = 0.0
        self.exchange_fields_calls = 0
        self.exchange_particles_calls = 0

    def step(self, state, current: float, dt: float):
        self.energy += dt
        r = self.radii[self.calls]
        self.calls += 1
        return r, self.energy, current

    def exchange_fields(self):
        self.exchange_fields_calls += 1
        zero = np.zeros(1)
        return (zero, zero, zero), (zero, zero, zero)

    def exchange_particles(self):
        self.exchange_particles_calls += 1
        return np.zeros((0, 3)), np.zeros((0, 3))


class DummySolver:
    def step(self, state, dt, current=0.0):
        return state


def test_hybrid_pic_feedback(monkeypatch):
    monkeypatch.setattr("dpf2.pinch_models.HallMHDSolver", DummySolver)
    pic = DummyPIC()
    model = HybridPinchModel(pic_driver=pic, switch_radius=1.0)
    t = [0.0, 1.0, 2.0]
    I = [0.0, 0.0, 0.0]
    res = model.run(t, I)
    assert np.allclose(res.radius, [0.1, 0.2, 0.3])
    assert res.energy[1] - res.energy[0] == pytest.approx(1.0)
    assert res.energy[2] - res.energy[1] == pytest.approx(1.0)
    assert pic.exchange_fields_calls == 3
    assert pic.exchange_particles_calls == 3
