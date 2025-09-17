"""Simple Bayesian parameter inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class ParameterEstimate:
    """Gaussian estimate of a scalar parameter."""

    mean: float
    variance: float


class BayesianParameterInference:
    """Perform lightweight Bayesian updates for simulation parameters.

    The implementation assumes independent Gaussian priors for each
    parameter and a user supplied ``model`` that maps a parameter dictionary
    to predicted diagnostic values.  The :meth:`update` method then performs a
    Kalman-style update for any diagnostics provided, returning the updated
    parameter means.
    """

    def __init__(
        self,
        parameters: Dict[str, ParameterEstimate],
        model: Callable[[Dict[str, float]], Dict[str, float]],
    ) -> None:
        self.parameters = parameters
        self.model = model

    # ------------------------------------------------------------------
    def update(
        self,
        diagnostics: Dict[str, float],
        noise: Dict[str, float],
    ) -> Dict[str, float]:
        """Update parameter estimates using experimental diagnostics.

        Parameters
        ----------
        diagnostics:
            Mapping of diagnostic names to measured values.
        noise:
            Mapping of diagnostic names to measurement variances.

        Returns
        -------
        Dict[str, float]
            Updated parameter means.
        """

        # Evaluate the forward model at the current parameter means
        param_means = {name: p.mean for name, p in self.parameters.items()}
        prediction = self.model(param_means)

        for name, obs in diagnostics.items():
            if name not in prediction or name not in self.parameters:
                continue

            pred = prediction[name]
            param = self.parameters[name]
            meas_var = noise.get(name, 1.0)

            # Kalman gain for scalar update
            gain = param.variance / (param.variance + meas_var)
            param.mean = pred + gain * (obs - pred)
            param.variance = (1.0 - gain) * param.variance

        return {name: p.mean for name, p in self.parameters.items()}


__all__ = ["BayesianParameterInference", "ParameterEstimate"]
