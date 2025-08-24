from pathlib import Path

import numpy as np

from dpf2.validation_suite import compute_error_metrics


def dataset_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "Validation"


def test_validation_passes_for_close_profiles():
    sim_outputs = {
        "gv_time_us": 2.6,
        "I": (np.array([0, 1, 2, 3, 4, 5]), np.array([0, 11, 19, 31, 39, 52])),
        "L": (np.array([0, 1, 2, 3, 4, 5]), np.array([10, 10.5, 12.5, 13, 14.5, 15.5])),
    }
    tolerances = {"gv_timing_us": 0.2, "I(t)": 5.0, "L(t)": 2.0}
    metrics = compute_error_metrics(sim_outputs, dataset_dir(), tolerances)
    assert metrics["passed"]
    assert metrics["gv_timing_us"] <= 0.2
    assert metrics["I_rmse"] <= 5.0
    assert metrics["L_rmse"] <= 2.0


def test_validation_fails_when_out_of_bounds():
    sim_outputs = {
        "gv_time_us": 5.0,
        "I": (np.array([0, 1, 2, 3, 4, 5]), np.zeros(6)),
        "L": (np.array([0, 1, 2, 3, 4, 5]), np.zeros(6)),
    }
    tolerances = {"gv_timing_us": 0.2, "I(t)": 5.0, "L(t)": 2.0}
    metrics = compute_error_metrics(sim_outputs, dataset_dir(), tolerances)
    assert not metrics["passed"]
