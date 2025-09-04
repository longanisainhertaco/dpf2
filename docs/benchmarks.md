# Benchmarks

DPF2 ships with both frozen regression benchmarks and small analytic
examples that exercise individual physics models.  These benchmarks act as
guard rails against code regression and provide usage examples for the
simulation infrastructure.

## Frozen regression suite

Predefined projects live under `benchmarks/`.  Each project contains two files:

- `deck.json` – configuration passed to the simulator
- `reference.csv` – reference time histories for key signals

Run a single case and produce a pass/fail dashboard plus an overlay plot using:

```bash
dpf2 benchmark run UNU
```

The command writes plots showing the simulation output overlaid on the
reference trace and prints whether the waveform falls within tolerance.  By
default results are written to `Validation/<case>/` where the overlay,
`metrics.json` and `results.h5` manifest are stored.

The regression suite ships with frozen inputs and reference outputs for three
devices:

- `UNU` – UNU/ICTP Plasma Focus Facility
- `PF-1000` – PF‑1000 device
- `MJOLNIR` – MJOLNIR dense plasma focus

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
