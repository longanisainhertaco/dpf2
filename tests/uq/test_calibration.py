import random
import numpy as np
import pytest

from dpf2.uq.calibration import (
    bayesian_calibration,
    nested_calibration,
    emcee_calibrate,
    dynesty_calibrate,
)


def _linear_model(params):
    a = params[0]
    x = np.linspace(0, 1, 10)
    return a * x


def test_bayesian_and_nested_calibration():
    rng = random.Random(0)
    true_a = 2.0
    x = np.linspace(0, 1, 10)
    noise = np.array([rng.gauss(0, 0.01) for _ in x])
    data = true_a * x + noise
    bounds = {"a": (0.0, 4.0)}

    samples = bayesian_calibration(_linear_model, bounds, data, n_samples=200, seed=0)
    mean_a = float(np.mean(samples["a"]))
    assert abs(mean_a - true_a) < 0.5

    ns = nested_calibration(_linear_model, bounds, data, n_live=20, n_iter=100, seed=0)
    mean_ns = float(np.mean(ns["a"]))
    assert abs(mean_ns - true_a) < 0.5


def test_emcee_and_dynesty_calibration():
    pytest.importorskip("emcee")
    pytest.importorskip("dynesty")
    rng = random.Random(1)
    true_a = 1.5
    x = np.linspace(0, 1, 10)
    noise = np.array([rng.gauss(0, 0.01) for _ in x])
    data = true_a * x + noise
    bounds = {"a": (0.0, 3.0)}

    emcee_samples = emcee_calibrate(
        _linear_model, bounds, data, n_walkers=8, n_steps=40, seed=0
    )
    mean_emcee = float(np.mean(emcee_samples["a"]))
    assert abs(mean_emcee - true_a) < 0.3

    dynesty_samples = dynesty_calibrate(
        _linear_model, bounds, data, n_live=20, n_iter=100, seed=0
    )
    mean_dynesty = float(np.mean(dynesty_samples["a"]))
    assert abs(mean_dynesty - true_a) < 0.3
