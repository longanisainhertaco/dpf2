"""Optimization utilities for parameter inference and control."""

import warnings


class OptimizationWarning(RuntimeWarning):
    """Warning raised when queries fall outside the trained domain."""


def enable_optimization_warning_as_error() -> None:
    """Escalate :class:`OptimizationWarning` to an exception.

    Optimisation routines may call this to ensure that out-of-distribution
    queries halt the search rather than silently producing invalid results.
    """

    warnings.filterwarnings("error", category=OptimizationWarning)


from .bayesian import BayesianParameterInference, ParameterEstimate
from .param_sweep import run_parametric_sweep
from .multi_objective import random_pareto_search

__all__ = [
    "BayesianParameterInference",
    "ParameterEstimate",
    "run_parametric_sweep",
    "random_pareto_search",
    "OptimizationWarning",
    "enable_optimization_warning_as_error",
]

