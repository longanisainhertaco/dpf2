# Surrogate Models

This project ships lightweight surrogate models for neutron yield and pinch time. The models are trained on benchmark simulation data under `data/benchmarks` using simple linear regression on peak discharge current.

## Training provenance

1. Collect peak current, neutron yield and pinch time from each benchmark case.
2. Fit a linear model `y = a*x + b` for each target.
3. Estimate residuals and compute the 95% conformal quantile to provide prediction intervals.
4. Record the training domain, feature mean and variance for out-of-distribution checks.

## Limits

The surrogates are valid only within the observed training range and for parameter combinations similar to the benchmark set. During inference the model computes a Mahalanobis distance from the training mean and raises an error when the input lies outside the 2σ (≈95%) region or outside the recorded training domain. These intervals quantify model uncertainty but do not replace high‑fidelity simulations.

