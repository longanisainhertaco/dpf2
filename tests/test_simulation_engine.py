import numpy as np

from dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine


def test_engine_runs():
    cfg = DPFConfig.with_defaults()
    cfg = cfg.model_copy(update={"simulation_control": cfg.simulation_control.model_copy(update={"time_end": 1e-6})})
    engine = SimulationEngine(cfg)
    results = engine.run()
    # basic shape checks
    assert results.current.size > 0
    assert results.time.shape == results.current.shape
    assert results.radius.shape == results.current.shape
    assert results.temperature.shape == results.current.shape
    assert results.pressure.shape == results.current.shape

    # time should start at zero and be monotonically increasing
    assert np.isclose(results.time[0], 0.0)
    assert np.all(np.diff(results.time) > 0)
    assert results.time[-1] <= 1e-6

    # physical quantities should be finite and non-negative
    assert np.all(np.isfinite(results.temperature))
    assert np.all(results.temperature >= 0)
    assert np.all(results.pressure >= 0)
    assert np.all(results.radius >= 0)

    # neutron yield should be non-negative
    assert results.neutron_yield >= 0.0

    # simple oscillation check in current waveform
    assert np.any(np.diff(np.sign(results.current)) != 0)

    # ensure conversion to dictionary preserves keys
    as_dict = results.to_dict()
    for key in ["time", "current", "pinch_radius", "temperature", "pressure", "neutron_yield"]:
        assert key in as_dict
