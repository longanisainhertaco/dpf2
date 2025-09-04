# Benchmarks

DPF2 ships with both frozen regression benchmarks and small analytic
examples that exercise individual physics models.  These benchmarks act as
guard rails against code regression and provide usage examples for the
simulation infrastructure.

## Frozen regression suite

Predefined projects live under `benchmarks/`.  Each project contains two files:

- `inputs.json` – configuration passed to the simulator
- `expected.json` – reference time histories with tolerance bands

Run a single case and produce a pass/fail dashboard plus an overlay plot using:

```bash
dpf2 run-benchmark unu_pff
```

To execute the entire suite at once:

```bash
dpf2 run-compare
```

Both commands write plots showing the simulation output overlaid on grey
tolerance bands and print a summary of whether current, voltage and neutron
yield fall within their respective thresholds.  Results for `run-benchmark`
are written to `Validation/<case>/` by default where the overlay and metrics
are stored.

The regression suite ships with frozen inputs and reference outputs for three
devices:

- `unu_pff` – UNU/ICTP Plasma Focus Facility
- `pf_1000` – PF‑1000 device
- `mjolnir` – MJOLNIR dense plasma focus

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
