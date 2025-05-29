import numpy as np

from dpf2.eos import RealGasEOS
from dpf2.fusion import bosch_hale_dd
from dpf2.pinch_models import SemiAnalyticPinchModel


def test_real_gas_cp_and_pressure():
    eos = RealGasEOS(gamma=1.4, cp0=1000.0, cp_slope=0.5)
    T = 300.0
    rho = 0.1
    cp = eos.cp(T)
    assert cp > 1000.0
    p = eos.pressure(rho, T)
    assert p > 0.0


def test_bosch_hale_positive():
    r = bosch_hale_dd(10.0)
    assert r > 0.0


def test_semi_analytic_yield_positive():
    model = SemiAnalyticPinchModel()
    t = np.linspace(0, 1e-6, 10)
    I = np.ones_like(t) * 1e4
    res = model.run(t, I)
    assert res.neutron_yield >= 0.0
