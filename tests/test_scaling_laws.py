import pytest
from dpf2.core.config import DPFConfig
from dpf2.scaling_laws import sweep_yield_scaling


def test_current_scaling_exponent():
    cfg = DPFConfig()
    res = sweep_yield_scaling(cfg, 'charging_voltage', [10000.0, 15000.0, 20000.0])
    assert res['m_current'] == pytest.approx(8.0, rel=0.2)


def test_pressure_scaling_exponent():
    cfg = DPFConfig()
    res = sweep_yield_scaling(cfg, 'initial_pressure', [80.0, 133.3, 200.0])
    assert res['m_current'] == pytest.approx(8.0, rel=0.2)
    assert res['m_parameter'] == pytest.approx(-4.0, rel=0.2)
