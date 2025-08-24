# Tabular EOS and Ionization

Integrate tabulated equation-of-state (EOS) tables and collisional ionization
models into the simulation framework.

## Expected Inputs
- Configuration referencing at least one external EOS table.
- Test states (density, temperature) covering the table range.

## Expected Outputs
- Pressure and internal energy values interpolated from the table.
- Ionization fraction for each test state.

## Acceptance Thresholds
- Interpolated pressure and energy within 1% of reference table values for the
  test states.
- Ionization predictions within 5% of a trusted collisional-radiative model.

## Demonstration
Provide a recorded demo or notebook evaluating the EOS/ionization module on the
specified test states and comparing to reference data. Link the demonstration in
the pull request.
