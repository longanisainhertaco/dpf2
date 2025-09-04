# Run Benchmark

The `benchmark run` command executes a single frozen benchmark
configuration and compares the results against reference outputs stored
under `benchmarks/<case>`.

Each benchmark directory contains two files:

- `deck.json` – simulation configuration passed to `DPFSimulation`.
- `reference.csv` – reference time histories for key signals.

Running the command produces a pass/fail dashboard and an overlay plot for
the three diagnostics. By default results are written to
`Validation/<case>/` where both the plot (`overlay.png`) and a
`metrics.json` summary of errors plus an HDF5 manifest are stored:

```bash
$ dpf2 benchmark run UNU
```

The generated plot overlays the reference trace and simulation results.
`metrics.json` captures RMS error statistics while `results.h5` records
the code hash and configuration digest alongside the raw waveform.
