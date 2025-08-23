import numpy as np
from circuit_config import CircuitConfig
from rlc_solver import run_circuit_simulation


def test_rlc_solver_runs():
    cfg = CircuitConfig.with_defaults()
    t, i, v, im, vm = run_circuit_simulation(cfg, t_end=10.0)
    assert len(t) == len(i) == len(v) == len(im) == len(vm)
    assert np.isclose(i[0], 0.0)
    assert v[0] > v[-1]


def test_linear_plasma_inductance():
    cfg = CircuitConfig.with_defaults().model_copy(
        update={
            "switch_delay": 0.0,
            "R_ext": 0.0,
            "C_ext": 1e9,
            "plasma_inductance_profile": [(0.0, 0.0), (10.0, 1.0)],
        }
    )
    t, i, v, im, vm = run_circuit_simulation(cfg, t_end=10.0)
    L_ext = cfg.L_ext * 1e-6
    V0 = cfg.V0 * 1e3
    expected = V0 * t / (L_ext + 0.1 * t)
    assert np.allclose(i, expected, rtol=1e-2, atol=1e-3)


def test_mutual_inductance_drive():
    cfg = CircuitConfig.with_defaults().model_copy(
        update={
            "switch_delay": 0.0,
            "R_ext": 0.0,
            "C_ext": 1e9,
            "V0": 0.0,
            "mutual_inductance_profile": [(0.0, 0.5), (10.0, 0.5)],
            "mutual_current_profile": [(0.0, 0.0), (10.0, 10.0)],
        }
    )
    t, i, v, im, vm = run_circuit_simulation(cfg, t_end=10.0)
    L_ext = cfg.L_ext * 1e-6
    M = 0.5e-6
    expected = -(M / L_ext) * im
    assert np.allclose(i, expected, rtol=1e-2, atol=1e-3)
