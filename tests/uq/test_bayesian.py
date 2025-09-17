import numpy as np
import pytest
import math

from dpf2.uq.calibration import bayes_factor, posterior_summary


def test_bayes_factor():
    bf = bayes_factor(-5.0, -6.0)
    assert bf == pytest.approx(math.exp(1.0))


def test_posterior_summary():
    samples = {"a": np.array([1.0, 2.0, 3.0, 4.0])}
    stats = posterior_summary(samples)
    assert pytest.approx(stats["a"]["mean"], rel=1e-6) == 2.5
    assert stats["a"]["std"] > 0
    assert stats["a"]["lower"] < stats["a"]["upper"]

