import numpy as np
from dpf2.pinch_models import AnalyticPinchModel, SemiAnalyticPinchModel


def test_analytic_model():
    model = AnalyticPinchModel()
    t = np.linspace(0, 1e-6, 10)
    I = np.ones_like(t) * 1e4
    res = model.run(t, I)
    assert res.radius.size == t.size
    assert res.neutron_yield >= 0.0


def test_semi_analytic_model():
    model = SemiAnalyticPinchModel()
    t = np.linspace(0, 1e-6, 10)
    I = np.ones_like(t) * 1e4
    res = model.run(t, I)
    assert res.radius.size == t.size
    assert res.axial_position is not None
