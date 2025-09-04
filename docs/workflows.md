# Workflows

This section demonstrates common workflows for calibrating simulations
against laboratory diagnostics and exploring trade-offs between neutron
yield and spot size.

## Bayesian and Nested Calibration

The `dpf2.uq.calibration` module provides both a Metropolis-Hastings MCMC
routine and a lightweight nested sampler.  The script below illustrates
calibration of a simple linear model against synthetic data:

```python
python docs/examples/calibration_example.py
```

## Multi-objective Optimization

To study yield versus spot size trade-offs, the `random_pareto_search`
function performs a random search and returns the estimated Pareto front.
An example run is provided in:

```python
python docs/examples/optimization_example.py
```

