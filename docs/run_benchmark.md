# Run Benchmark

The `run-benchmark` command executes a single frozen benchmark
configuration and compares the results against reference outputs stored
under `benchmarks/<case>`.

Each benchmark directory contains two files:

- `inputs.json` – simulation configuration passed to `DPFSimulation`.
- `expected.json` – reference time histories and tolerance bands for
  current, voltage and neutron yield.

Running the command produces a pass/fail dashboard and an overlay plot for
the three diagnostics. By default results are written to
`Validation/<case>/` where both the plot (`overlay.png`) and a
`metrics.json` summary of errors are stored:

```bash
$ dpf2 run-benchmark unu_pff
```

The generated plot includes grey tolerance bands around the reference
traces with the actual simulation output overlaid.  A table lists whether
current, voltage and neutron yield fall within their respective bands, and
the metrics file records both the maximum and RMS deviation for each
signal.
