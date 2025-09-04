import random
import numpy as np
import pytest

from dpf2.uq.inference import emcee_infer


def test_emcee_infer_linear_model():
    pytest.importorskip("emcee")
    rng = random.Random(0)

    def model(params):
        a = params[0]
        x = np.linspace(0, 1, 5)
        return a * x

    true_a = 2.0
    x = np.linspace(0, 1, 5)
    noise = np.array([rng.gauss(0, 0.01) for _ in x])
    data = true_a * x + noise

    samples = emcee_infer(
        model,
        {"a": (0.0, 4.0)},
        data,
        n_walkers=6,
        n_steps=20,
        sigma=0.01,
        seed=0,
    )
    mean_a = float(np.mean(samples["a"]))
    assert abs(mean_a - true_a) < 0.5
