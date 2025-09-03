import pytest

from dpf2.pinch_models import HybridPinchModel
from dpf2.physics.pic_driver import PicDriver


class DummyPIC(PicDriver):
    def __init__(self):
        self.calls = []
        self.radii = [2.0, 0.5, 2.0]
        self.energy = 0.0
        self._idx = 0

    def step(self, current: float, dt: float):
        self.calls.append((current, dt))
        self.energy += dt
        radius = self.radii[self._idx]
        self._idx += 1
        return radius, self.energy


class DummySolver:
    def __init__(self):
        self.calls = 0

    def step(self, state, dt, current=0.0):
        self.calls += 1
        if self.calls == 1:
            state.rho[...] = 0.0
            state.rho[state.rho.shape[0] // 2, state.rho.shape[1] // 2, :] = 1.0
        elif self.calls == 2:
            state.rho[...] = 1.0
        return state


def test_hybrid_transition(monkeypatch):
    monkeypatch.setattr("dpf2.pinch_models.HallMHDSolver", DummySolver)
    pic = DummyPIC()
    model = HybridPinchModel(pic_driver=pic, switch_radius=1.0)
    t = [0.0, 1e-7, 2e-7]
    I = [0.0, 0.0, 0.0]
    res = model.run(t, I)
    assert res.radius[0] > 1.0
    assert res.radius[1] == pytest.approx(0.5)
    assert res.radius[2] > 1.0
    assert pic.calls == [(0.0, 0.0), (0.0, 1e-7), (0.0, 1e-7)]
