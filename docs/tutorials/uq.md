# Uncertainty Quantification and Calibration

DPF2 provides sampling utilities to explore parameter space and Bayesian
calibration tools to infer model parameters from diagnostic
measurements.

## Sampling sweeps

Latin hypercube and Sobol sequence samplers generate low discrepancy
sets of parameters. Use the command line helpers to create batches for
large sweeps:

```
dpf2 latin-hypercube --parameters '{"capacitance":[1e-6,5e-6]}' --samples 8
```

```
dpf2 sobol-sample --parameters '{"capacitance":[1e-6,5e-6]}' --samples 8
```

The commands write JSON arrays where each entry contains a mapping of
parameter names to sampled values. These files can be consumed by batch
systems to launch many simulations in parallel.

## Bayesian calibration

The :func:`dpf2.uq.calibration.bayesian_calibration` routine implements a
simple Metropolis-Hastings sampler. Provide a model function that accepts
an array of parameters and returns predictions aligned with your
measurements:

```python
from dpf2.uq.calibration import bayesian_calibration

def model(params: np.ndarray) -> np.ndarray:
    # return diagnostic predictions for given parameters
    ...

posterior = bayesian_calibration(model, bounds, data)
```

Posterior samples for each parameter are returned as NumPy arrays.

## Notebooks

Example notebooks demonstrating these workflows are available in
``examples/notebooks``:

- ``uq_sampling.ipynb`` – generating Latin hypercube and Sobol samples
- ``bayesian_calibration.ipynb`` – calibrating a toy model using MCMC

