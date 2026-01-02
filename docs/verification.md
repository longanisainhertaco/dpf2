# Verification Tests

This guide summarises common test problems used to verify the numerical implementation.

## Quick numerics verification (CLI/GUI)

Run the lightweight numerics checks directly from the CLI:

```bash
python -m dpf2.cli.main verify-numerics --sizes 16 --sizes 32 --sizes 64 \
    --output synthetic_diagnostics/verification.h5
```

The command executes the Brio–Wu, Orszag–Tang and MMS problems at the requested
grid sizes, writes the metrics to the HDF5 file, and prints the observed order
of convergence for each problem.  A healthy solver should report an observed
order close to the design value of 1.0 or higher.

For GUI-driven workflows, :class:`dpf2.ui.verification_panel.VerificationPanelUI`
wraps the same routines and provides a ``summarize()`` helper that returns a
human-readable report for dashboards and notebooks.

## Method of Manufactured Solutions (MMS)
1. Select an analytic solution for the governing equations.
2. Compute the required source terms and boundary conditions so the analytic fields satisfy the modified system.
3. Run the simulation with the sources enabled and compare the numerical solution against the analytic expression.
4. Verify that the error decreases at the expected order with grid refinement.

## Brio--Wu Shock Tube
1. Initialise left and right states of a 1-D MHD shock tube with discontinuous magnetic field and density.
2. Evolve to the reference time and compare density and magnetic field profiles with the published solution.
3. This test exercises the Riemann solver and divergence control algorithm.

## Orszag--Tang Vortex
1. Start from a doubly periodic domain with sinusoidal velocity and magnetic fields.
2. Track the development of MHD turbulence and compare energy spectra or time histories of peak current density.
3. Used to assess robustness of shock capturing and resolution of non-linear interactions.

## Benchmarks with tolerance bands

Curated dense-plasma-focus datasets ship under ``benchmarks/`` with expected
traces and tolerance bands.  In addition to the frozen PF1000 and UNU examples,
the suite now includes:

- ``gv_trajectory`` – inductive drive reproducing the Gratton–Vargas trajectory
- ``inductance_overlay`` – overlaid current/voltage traces emphasising inductive swing

Validate against any benchmark using the helper script:

```bash
python scripts/run_benchmark.py gv_trajectory
python scripts/run_benchmark.py inductance_overlay
```

Each run writes ``metrics.json``, ``overlay.png`` and an HDF5 manifest under
``Validation/<case>/``.  The report grades each signal using the supplied
tolerance bands; grades A or B indicate agreement within tolerance.

## References
1. C. J. Roy, "Review of code and solution verification procedures for computational simulation," J. Comput. Phys., 2010.
2. M. Brio and C. C. Wu, "An Upwind Differencing Scheme for the Equations of Ideal Magnetohydrodynamics," J. Comput. Phys., 1988.
3. S. A. Orszag and C. M. Tang, "Small-scale structure of two-dimensional magnetohydrodynamic turbulence," J. Fluid Mech., 1979.
