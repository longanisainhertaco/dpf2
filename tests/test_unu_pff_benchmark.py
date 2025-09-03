import importlib


def test_unu_pff_benchmark_runs():
    module = importlib.import_module("examples.benchmarks.unu_pff_benchmark")
    metrics = module.run_benchmark()
    assert "pinch_time_error" in metrics
    assert metrics["pinch_time_error"] >= 0.0
