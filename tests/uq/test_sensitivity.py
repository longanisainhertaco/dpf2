import pytest

from dpf2.uq.analysis import sobol_indices, propagate_jitter_voltage_pressure


def test_sobol_indices_linear():
    x = [i / 9 for i in range(10)]
    y = [2 * xi + 1 for xi in x]
    samples = [[xi] for xi in x]
    res = sobol_indices(samples, y, ["x"])
    assert res["x"] == pytest.approx(1.0)


def test_propagate_jitter_voltage_pressure():
    def model(params):
        v, p = params
        return v + p, v - p

    stats = propagate_jitter_voltage_pressure(
        model,
        voltage=100.0,
        pressure=10.0,
        voltage_jitter_pct=0.1,
        pressure_jitter_pct=0.2,
        n_samples=100,
        alpha=0.9,
        seed=0,
    )
    assert stats["neutron_yield"]["std"] > 0.0
    assert stats["pinch_time"]["std"] > 0.0

