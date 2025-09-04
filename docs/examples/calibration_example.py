import numpy as np

from dpf2.uq import bayesian_calibration, nested_calibration


def model(params):
    a = params[0]
    x = np.linspace(0, 1, 20)
    return a * x


true_a = 1.2
x = np.linspace(0, 1, 20)
noise = np.random.default_rng(0).normal(0, 0.01, size=x.shape)
data = true_a * x + noise

bounds = {"a": (0.0, 2.0)}

mcmc_samples = bayesian_calibration(model, bounds, data, n_samples=500, seed=0)
nested_samples = nested_calibration(model, bounds, data, n_live=20, n_iter=200, seed=0)

print("MCMC mean:", float(np.mean(mcmc_samples["a"])))
print("Nested mean:", float(np.mean(nested_samples["a"])))

