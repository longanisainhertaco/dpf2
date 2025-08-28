"""Example closed-loop optimisation pipeline for digital twin workflows.

This script demonstrates how experimental diagnostics can be streamed and
compared against simulation predictions to update model parameters in real
time.  The workflow is intentionally lightweight and operates on the simple
``CouplingState`` object used throughout the code base.
"""

from __future__ import annotations

from dpf2.core.bases import CouplingState
from dpf2.diagnostics.streaming import NeutronYieldStreamer, RealTimeComparator
from dpf2.optimization import BayesianParameterInference, ParameterEstimate


def forward_model(params: dict[str, float]) -> dict[str, float]:
    """Minimal forward model linking parameters to diagnostics."""

    # Here we assume the neutron yield scales with the square of a
    # ``current_scale`` parameter for demonstration purposes.
    scale = params["current_scale"]
    return {"neutron_yield": 1.0e5 * scale ** 2}


def main() -> None:
    # -- Prior parameter estimate ----------------------------------------
    inference = BayesianParameterInference(
        {"current_scale": ParameterEstimate(mean=1.0, variance=0.5)},
        forward_model,
    )

    # -- Set up streaming diagnostics and comparison hooks ---------------
    def comparison_callback(t: float, sim: float, exp: float) -> None:
        print(f"t={t:.2e}s  sim={sim:.2e}  exp={exp:.2e}  residual={exp - sim:.2e}")

    comparator = RealTimeComparator(comparison_callback)
    streamer = NeutronYieldStreamer(callback=lambda t, v: comparator.compare(t, v), comparator=comparator)

    # Inject a synthetic experimental measurement at t=1 microsecond
    comparator.ingest(1e-6, 5.0e5)

    # -- Run a toy simulation step ---------------------------------------
    state = CouplingState(current=2.0, voltage=0.0)
    streamer.record(state, 1e-6)

    # -- Update parameters using the measurement -------------------------
    updated = inference.update({"neutron_yield": 5.0e5}, {"neutron_yield": 1.0e4})
    print("Updated parameter means:", updated)


if __name__ == "__main__":
    main()

