import importlib


def test_analytic_plasma_expansion_runs():
    module = importlib.import_module("benchmarks.analytic_plasma_expansion")
    error = module.run_benchmark()
    assert error == 0.0


def test_bohm_sheath_benchmark_runs():
    module = importlib.import_module("benchmarks.bohm_sheath_benchmark")
    errors = module.run_benchmark()
    assert errors["field_error"] < 1e-12
    assert errors["velocity_error"] < 1e-12
