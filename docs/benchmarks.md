# Benchmarks

DPF2 ships with both frozen regression benchmarks and small analytic
examples that exercise individual physics models.  These benchmarks act as
guard rails against code regression and provide usage examples for the
simulation infrastructure.

## Frozen regression suite

Predefined projects live under `benchmarks/`.  Each reference device has a
read‑only directory containing three JSON files:

- `deck.json` – minimal simulation configuration for the reference shot
- `inputs.json` – waveform and scalar outputs produced by the simulator
- `expected.json` – published reference data used for validation

Run a benchmark using the CLI ``validate`` command:

```bash
python -m dpf2.cli.main validate \
    --config benchmarks/PF1000/deck.json \
    --dataset benchmarks/PF1000 \
    --outdir Validation/PF1000
```

The command compares the current trace, neutron yield, and anisotropy against
``expected.json`` and writes a ``benchmark_report.json`` summarising pass/fail
status for each metric.

### Acceptance tolerances

- Current trace: RMSE within 10 % of the peak reference current
- Neutron yield: 5 % relative error
- Anisotropy: 5 % relative error

### Reference devices

- `PF1000` – PF‑1000 facility [Sadowski 1992]
- `UNU` – UNU/ICTP Plasma Focus Facility [Lee 2014]
- `MJOLNIR` – LLNL MJOLNIR dense plasma focus [Eddleman 2020]

## Analytic plasma expansion

`benchmarks/analytic_plasma_expansion.py` models the expansion of a
quasi-neutral plasma into vacuum using the isothermal approximation.  The ion
front position is

$$R(t) = R_0 + c_s t,$$

where $c_s$ is the ion sound speed.  See Spitzer (1962) and Dawson (1960) for
details.  Run the benchmark with

```bash
python -m benchmarks.analytic_plasma_expansion
```

## Bohm sheath formation

`benchmarks/bohm_sheath_benchmark.py` exercises the `BohmSheath` class and
verifies the expected sheath potential drop and Bohm velocity
(see Bohm 1949).  Execute with

```bash
python -m benchmarks.bohm_sheath_benchmark
```

## Continuous integration

The benchmark scripts can be incorporated into continuous integration as
long-running jobs or nightly builds.  The tests in `tests/test_benchmarks.py`
provide a lightweight example that can be enabled in CI pipelines.
