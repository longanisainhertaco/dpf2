import numpy as np

from dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine


def test_engine_runs():
    cfg = DPFConfig.with_defaults()
    cfg = cfg.model_copy(update={"simulation_control": cfg.simulation_control.model_copy(update={"time_end": 1e-6})})
    engine = SimulationEngine(cfg)
    results = engine.run()
    assert results.current.size > 0
    assert results.neutron_yield >= 0.0
    # simple oscillation check
    assert np.any(np.diff(np.sign(results.current)) != 0)
