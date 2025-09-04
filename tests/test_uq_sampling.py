import numpy as np

from dpf2.uq.sampling import latin_hypercube, sobol_sample


def test_latin_hypercube_bounds():
    bounds = {"a": (0.0, 1.0), "b": (-1.0, 1.0)}
    samples = latin_hypercube(bounds, 4, seed=1)
    assert samples.shape == (4, 2)
    assert all(0 <= s <= 1 for s in samples[:, 0])
    assert all(-1 <= s <= 1 for s in samples[:, 1])


def test_sobol_sample_bounds():
    bounds = {"x": (0.0, 2.0)}
    samples = sobol_sample(bounds, 4, seed=2)
    assert samples.shape == (4, 1)
    assert all(0 <= s <= 2 for s in samples[:, 0])
