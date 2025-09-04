# Physics Models

## Hall-MHD

The Hall-MHD option augments the single-fluid MHD system with Hall, electron-pressure
gradient, and electron inertia terms.  Enable the model by setting
`hallMhdEnabled` in the physics configuration section.  Transport coefficients are
sourced from simplified Braginskii expressions using NRL Formulary lookups.

Activation is gated on two dimensionless parameters:

* the electron magnetisation \(\omega_{ce} \tau_e\).  The Hall terms become
  important when this value exceeds unity;
* the ratio of the ion inertial length to the system size \(d_i/L\).  Two-fluid
  effects are resolved when this ratio is larger than a few percent.

When both thresholds are satisfied the solver switches to a two-fluid regime,
which captures whistler dispersion and Hall shocks.  These features are most
relevant in strongly magnetised, low-density plasmas where small spatial scales
approach the ion inertial length.
