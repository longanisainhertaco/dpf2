import importlib

from diagnostics.neutron.benchmarks import (
    load_pf1000_reference,
    load_mjolnir_reference,
    within_pass_band,
    evaluate_pass_fail,
)


def test_reference_loading():
    pf = load_pf1000_reference()
    mj = load_mjolnir_reference()
    assert list(pf.columns) == ["time", "current"]
    assert list(mj.columns) == ["time", "current"]


def test_pass_band_evaluation():
    reference = [0.0, 10.0]
    good = [0.0, 10.5]
    bad = [0.0, 12.0]
    assert within_pass_band(good, reference, 0.1).all()
    assert not evaluate_pass_fail(bad, reference, 0.1)


def test_example_workflow_runs():
    module = importlib.import_module("examples.diagnostics.run_neutron_benchmarks")
    assert module.run_pf1000_benchmark()
    assert not module.run_mjolnir_benchmark()
