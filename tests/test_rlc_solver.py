import numpy as np
from circuit_config import CircuitConfig
from rlc_solver import run_circuit_simulation


def test_rlc_solver_runs():
    cfg = CircuitConfig.with_defaults()
    t, i, v = run_circuit_simulation(cfg, t_end=10.0)
    assert len(t) == len(i) == len(v)
    # basic sanity: current should start at 0 and eventually decrease
    assert np.isclose(i[0], 0.0)
    assert v[0] > v[-1]
