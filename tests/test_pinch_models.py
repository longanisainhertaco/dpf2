import numpy as np
import pytest
from dpf2.pinch_models import AnalyticPinchModel, SemiAnalyticPinchModel, MHDPinchModel


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


def test_mhd_model_energy_conservation():
    model = MHDPinchModel(grid_shape=(4, 4, 4))
    t = np.linspace(0, 1e-7, 5)
    I = np.ones_like(t) * 1e4
    res = model.run(t, I)
    assert res.energy is not None
    assert np.isclose(res.energy[0], res.energy[-1], rtol=1e-3)


def test_mhd_model_yield_scaling():
    t = np.linspace(0, 1e-7, 5)
    I1 = np.ones_like(t) * 1e4
    I2 = np.ones_like(t) * 2e4
    model = MHDPinchModel(grid_shape=(4, 4, 4))
    res1 = model.run(t, I1)
    res2 = model.run(t, I2)
    assert res2.neutron_yield > res1.neutron_yield
    assert res2.neutron_yield == pytest.approx(res1.neutron_yield * 4, rel=0.5)
