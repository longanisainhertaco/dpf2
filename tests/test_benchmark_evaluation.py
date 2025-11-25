import math

from dpf2.validation_suite import evaluate_benchmark


def test_evaluate_benchmark_with_slope_and_pinch():
    expected = {
        "current": {"time": [0.0, 1e-6, 2e-6], "value": [0.0, 1.0, 0.5]},
        "pinch_time": 1.5e-6,
        "neutron_yield": 1.0e9,
        "tolerance": {
            "current": 0.2,
            "current_slope": 0.25,
            "pinch_time": 0.1,
            "neutron_yield": 0.1,
        },
    }
    sim = {
        "current": {"time": [0.0, 1e-6, 2e-6], "value": [0.0, 0.95, 0.55]},
        "pinch_time": 1.45e-6,
        "neutron_yield": 0.95e9,
    }

    report = evaluate_benchmark(sim, expected)
    assert report["passed"]
    assert report["checks"]["current"]
    assert report["checks"]["current_slope"]
    assert report["checks"]["pinch_time"]
    assert math.isfinite(report["current_slope_error"])


def test_evaluate_benchmark_handles_missing_optional_metrics():
    expected = {
        "current": {"time": [0.0, 1e-6], "value": [0.0, 1.0]},
        "tolerance": {"current": 0.5, "current_slope": 0.5},
    }
    sim = {"current": {"time": [0.0, 1e-6], "value": [0.0, 1.0]}}
    report = evaluate_benchmark(sim, expected)
    assert report["passed"]
    # Optional metrics should not fail the evaluation when absent
    assert report["checks"]["current_slope"]
    assert "pinch_time_error" in report
