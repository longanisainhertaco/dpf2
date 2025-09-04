# Hall magnetohydrodynamics

The `HallMHDSolver` extends the resistive MHD system with optional Hall and
electron-inertia corrections together with simple Braginskii transport
closures.  These additions are enabled automatically based on two runtime
metrics evaluated every time step:

* **Electron magnetisation** – the solver computes the product of the electron
  gyrofrequency and collision time, ``\omega_{ce}\tau_e``.  When this exceeds
  `hall_threshold` the Hall term is activated.
* **Ion skin depth ratio** – the ratio ``d_i/L`` of the ion skin depth to a user
  supplied macroscopic scale length.  When this exceeds `ei_threshold` an
  electron-inertia contribution is added to Ohm's law.

The most recent values of these quantities together with boolean flags
indicating whether the corrections were applied are exposed through the solver
attributes `last_wce_tau_e`, `last_di_over_L`, `hall_active` and
`electron_inertia_active`.

A Braginskii transport closure may be provided via the ``braginskii``
callable.  The helper :func:`dpf2.physics.hall_mhd.nrl_braginskii` implements a
light-weight approximation to the NRL formulary and returns the parallel
viscosity and thermal conductivity coefficients.

```python
from dpf2.hall_mhd_solver import HallMHDSolver, MHDState
from dpf2.physics.hall_mhd import nrl_braginskii

state = MHDState(...)
solver = HallMHDSolver(braginskii=nrl_braginskii)
solver.step(state, dt)
print(solver.hall_active, solver.electron_inertia_active)
```
