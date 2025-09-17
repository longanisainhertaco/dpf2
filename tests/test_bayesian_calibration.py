import random
import numpy as np

from dpf2.uq.calibration import bayesian_calibration, nested_calibration


def test_bayesian_calibration_linear_model():
    rng = random.Random(0)

    def model(params):
        a = params[0]
        x = np.linspace(0, 1, 10)
        return a * x

    true_a = 2.0
    x = np.linspace(0, 1, 10)
    noise = np.array([rng.gauss(0, 0.01) for _ in x])
    data = true_a * x + noise

    samples = bayesian_calibration(
        model, {"a": (0.0, 4.0)}, data, n_samples=500, seed=0
    )
    mean_a = sum(samples["a"]) / len(samples["a"])
    assert abs(mean_a - true_a) < 0.5


def test_nested_calibration_linear_model():
    rng = random.Random(0)

    def model(params):
        a = params[0]
        x = np.linspace(0, 1, 10)
        return a * x

    true_a = 2.0
    x = np.linspace(0, 1, 10)
    noise = np.array([rng.gauss(0, 0.01) for _ in x])
    data = true_a * x + noise

    samples = nested_calibration(
        model, {"a": (0.0, 4.0)}, data, n_live=20, n_iter=200, seed=0
    )
    mean_a = sum(samples["a"]) / len(samples["a"])
    assert abs(mean_a - true_a) < 0.5
