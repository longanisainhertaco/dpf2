# Dynamic Circuit Coupling

Extend the circuit model to accept time-varying inductance and back-EMF feedback
from the plasma and verify conservation of total energy.

## Expected Inputs
- Simulation configuration enabling circuit–plasma feedback with an inductance
  model that changes during the run.
- Reference analytic solution for current and voltage.

## Expected Outputs
- Time series of circuit current, voltage, and plasma inductance.
- Computed initial and final total energy (circuit + plasma).

## Acceptance Thresholds
- Energy difference between initial and final states < 1% of the initial energy.
- Simulated current trace within 10% of the analytic reference throughout the run.

## Demonstration
Provide a recorded demo or Jupyter notebook running the coupling example and
showing that the outputs satisfy the thresholds. Link the demonstration in the
associated pull request.
