# Resistive MHD Solver

Verify that a 2-D resistive MHD solver with HLLD fluxes and constrained transport
can stably evolve a rundown/pinch scenario.

## Expected Inputs
- Configuration describing a 2-D coaxial geometry and RLC circuit coupling.
- Initial plasma density and temperature profiles.
- Time step and simulation duration sufficient to capture rundown and pinch.

## Expected Outputs
- Time series of magnetic field, density, and current profiles.
- Diagnostic report of total system energy throughout the run.

## Acceptance Thresholds
- Relative magnetic divergence `|∇·B|/|B|` < 1e-6 at all times.
- Energy conservation error < 1% between start and end of the simulation.
- Simulation completes the full rundown without numerical instability.

## Demonstration
Provide a recorded demo or Jupyter notebook executing the solver with the
above configuration and showing that the outputs meet the thresholds.
Link the demo/notebook in the pull request implementing this feature.
