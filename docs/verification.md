# Verification Tests

This guide summarises common test problems used to verify the numerical implementation.

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

## References
1. C. J. Roy, "Review of code and solution verification procedures for computational simulation," J. Comput. Phys., 2010.
2. M. Brio and C. C. Wu, "An Upwind Differencing Scheme for the Equations of Ideal Magnetohydrodynamics," J. Comput. Phys., 1988.
3. S. A. Orszag and C. M. Tang, "Small-scale structure of two-dimensional magnetohydrodynamic turbulence," J. Fluid Mech., 1979.
