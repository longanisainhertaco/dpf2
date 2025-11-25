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

## Sheath and Circuit Coupling

Sheath advance is driven by a lumped circuit that couples the capacitor bank to
the plasma column. The UI multi-pane plots mirror three model observables:

- **Bank voltage** is integrated using a stiff ODE solver with adaptive
  timesteps that resolve the breakdown transient.
- **Discharge current** comes from the same circuit equations, with induced
  voltage from sheath motion fed back into the system inductance.
- **Sheath radius** evolves through a snowplow model corrected with pressure and
  inductive terms; the WebGL overlay visualises this radius in real time.

## Numerical Schemes

- **Temporal integration**: implicit/explicit Runge–Kutta pairs for the circuit
  and sheath advance with error control tuned to the Courant limit of the
  plasma column.
- **Flux reconstruction**: high-order finite volumes with slope limiting in
  Hall-MHD mode; fallback TVD schemes are used when the Regime Dashboard reports
  strongly collisional conditions.
- **Source term handling**: operator splitting isolates ionisation, radiation
  losses and resistive heating; stability is enforced through adaptive substeps
  whose thresholds are tied to dimensionless groups visible in the dashboard.

## Dimensionless Regime Monitoring

The live dashboard consumes \(S\), \(\beta\), \(M_A\), \(R_m\), Knudsen and
magnetisation metrics streamed from the solver. Threshold crossings trigger
warnings in the UI and can be used to automatically swap between fluid and
two-fluid kernels.

## Synthetic Visual Diagnostics

For tutorial runs without backend data, the frontend synthesises consistent
current, voltage and sheath trajectories. These signals follow the same
qualitative scaling laws as the physics models so users can rehearse workflows
before consuming compute time.
