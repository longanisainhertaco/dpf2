# Validation Workflow

This document describes how to validate the synthetic diagnostics in this
repository. Benchmark definitions and expected diagnostic outputs are stored in
`tests/benchmarks`.

## Running the Benchmarks

1. Install the package dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the benchmark tests:
   ```bash
   pytest tests/benchmarks/test_diagnostic_baselines.py
   ```
   The tests compute neutron yield, X-ray spectra, and scope traces for the
   provided cases and compare the results against the stored baselines. These
   tests run in continuous integration to ensure diagnostic calculations remain
   consistent with the reference outputs.

## Updating Baselines

To update or add a benchmark, edit or create the JSON files in
`tests/benchmarks` and ensure the expected outputs match the new results. Commit
these baseline files together with any code changes.

## Uncertainty Analysis

Monte-Carlo ensembles quantify shot-to-shot variability by perturbing key inputs such as bank voltage and gas composition. For each dataset in `ReferenceMaterial`, hundreds of realizations are drawn with the prescribed jitter to compute the mean and one-sigma spread of peak current and neutron yield. The validation suite re-simulates these ensembles and checks that measured values fall inside a ±2σ envelope, corresponding to roughly a 95% confidence bound on the diagnostics.

## Uncertainty Propagation

The validation workflow now supports Monte-Carlo exploration of experimental
uncertainties. The `ExperimentalVariabilityModel` describes jitter in capacitor
voltage, fill pressure, and geometric tolerances. The helper class
`MonteCarloVariability` draws random realizations of these quantities and feeds
them to the `SimulationEngine`. Statistics for current, pressure and neutron
yield are aggregated (mean and standard deviation) across the ensemble.

```python
from dpf2.dpf_config import DPFConfig
from dpf2.experimental_variability import ExperimentalVariabilityModel, MonteCarloVariability
from dpf2.simulation_engine import SimulationEngine

cfg = DPFConfig.with_defaults()
var_cfg = ExperimentalVariabilityModel.with_defaults().model_copy(update={
    "pressure_jitter_pct": 5,
    "per_field_distribution_params": {
        "capacitor_voltage": {"jitter_pct": 2.0},
        "cathode_gap_degrees": {"jitter_pct": 1.0},
    },
    "stochastic_run_id": 42,
})
variability = MonteCarloVariability(var_cfg, realizations=50)
engine = SimulationEngine(cfg)
ensemble = engine.run(variability=variability)

# ensemble.current_mean and ensemble.current_std bound the current trace
```

When plotting diagnostics, the mean trace may be surrounded with an uncertainty
band using the ±1σ envelope from the ensemble statistics. The same approach
applies to neutron yield time series, providing an intuitive visualization of
expected experimental scatter.

### Statistical Error Bands

Each benchmark stores the ensemble mean and one-sigma spread for peak current and neutron yield in `ReferenceMaterial` JSON files (for example, `shot_deuterium_20kV.json`). Monte-Carlo validation tests recompute these statistics and ensure experimental measurements fall inside the ±2σ envelope. When adding new benchmarks, record these statistics so the validation suite can enforce the error bands.

### Uncertainty Bounds

For time-dependent diagnostics the validation suite reports both the mean and
two-sigma spread derived from the Monte-Carlo ensemble. These bounds correspond
to an approximate 95% confidence interval and simulated traces are expected to
remain within this envelope.

## Parameter Inference

High-resolution experimental traces can be used to infer uncertain circuit or
plasma parameters. A typical workflow minimizes the L2 error between a measured
trace and a family of simulated traces generated while scanning the parameters
of interest. The optimal parameter set is returned along with a covariance
estimate derived from the ensemble statistics, providing uncertainty bounds on
the inferred quantities.

## Physics Regression Tests

### MHD Shock Tube

A Sod-type shock tube problem checks the resistive MHD module against the
analytic solution published by Park et al. in 2023
[ReferenceMaterial/Park_POP_2023.pdf].  The reference density profile at
$t=0.1$ is stored in `ReferenceMaterial/mhd_shock_tube.json` and the solver
output must reproduce it with an L1 error below **10%**.

### Hall-MHD Snowplow

The Hall-MHD solver is validated with a coaxial snowplow inductance comparison.
For a 1 A current and $(r_{\text{inner}}, r_{\text{outer}})$ of 1 and 2 cm, the
computed plasma inductance is expected to match the analytic Lee model
[ReferenceMaterial/Lee-paper.pdf].  The expected inductance value is recorded
in `ReferenceMaterial/hall_snowplow.json` and the numerical result must agree
within **5%**.

### Distributed Circuit Matrices

A single 1 m transmission line segment (1 µH/m, 1 Ω/m, 1 µF/m) with parasitic
contributions of 1 nH, 0.1 Ω and 2 nF is combined with a closed 1 mΩ switch.
The diagonal R, L and C matrices assembled from this setup are compared against
`ReferenceMaterial/distributed_circuit.json` and must agree within a relative
tolerance of **10⁻9**.

### ALEGRA Energy Deposition

An energy history generated with the ALEGRA code serves as a benchmark for
the resistive MHD module.  DPF2 recomputes the evolution for densities
decreasing linearly from 1 to 0.6 and pressures increasing from 1 to 2 at
times 0, 0.5 and 1 µs.  The reference energies are stored in
`ReferenceMaterial/alegra_reference.json` and the regression test in
`tests/validation/test_alegra_reference.py` requires agreement within a
relative tolerance of **10⁻9**.

### MACH2 Flux Validation

A single-cell state initialized to unit density, 0.1 x-velocity and a small
axial magnetic field is compared against fluxes produced by the MACH2 code.
The x-direction fluxes from MACH2 are tabulated in
`ReferenceMaterial/mach2_reference.json`.  The DPF2 implementation must
reproduce these fluxes within a relative tolerance of **10⁻9** via
`tests/validation/test_mach2_reference.py`.


## Coupled Benchmarks

### Coupled Current Trace

A zero-dimensional plasma model with a linearly increasing inductance is
explicitly coupled to a series RLC circuit. The capacitor is initially charged
to 1 kV and discharged for 1 µs with a 10 ns time step. The resulting current
waveform is stored in `ReferenceMaterial/coupled_current.json` and the
regression test in `tests/validation/test_coupled_current_trace.py` requires the
L1 error between the simulated and reference traces to remain below **1e-6**.

### Coupled Current and Voltage Traces

A 1 µs discharge of the same coupled circuit is also sampled every 10 ns to
provide reference time, current and voltage profiles. These series are recorded
in `ReferenceMaterial/coupled_traces.json`. The regression test in
`tests/validation/test_coupled_traces.py` recomputes the evolution and requires
agreement with the references to within a relative tolerance of **1e-9**.

### Z-Machine Experimental Traces

A unit-valued RLC discharge (L=1 H, R=1 Ω, C=1 F, V₀=1 V) provides reference
current and voltage traces along with synthetic plasma diagnostics computed as
\(p = 10^{-2} I^2\) and \(T = 300 + 10^{-3} I\).  The time series and the
integrated neutron yield are stored in `ReferenceMaterial/z_machine_traces.json`.
The regression test in `tests/validation/test_exp_traces.py` recomputes these
profiles and enforces agreement with the references to within a relative
tolerance of **1e-9**.

### Experimental Shot Trace

An additional experimental discharge with L=1 H, R=2 Ω, C=0.5 F and
1 V initial capacitor voltage is provided as an RLC benchmark.  Time,
current and voltage traces together with the integrated neutron yield are
recorded in `ReferenceMaterial/experimental_shot.json`.  The regression test
`tests/validation/test_experimental_shot.py` verifies the reproduction of
these series within a relative tolerance of **10⁻9**.

### High-Resolution Experimental Trace

A 1 µs discharge sampled at 10 ns provides a high‑resolution version of the
experimental shot. The dataset in
`ReferenceMaterial/experimental_shot_highres.json` stores time, current,
voltage, pressure, temperature and the integrated neutron yield. The regression
test `tests/validation/test_regression_highres.py` compares the solver output
against this reference with a relative tolerance of **10⁻9**.
