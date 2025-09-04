# Run Benchmark

The `run-benchmark` command executes a single frozen benchmark
configuration and compares the results against reference outputs stored
under `Reference/Benchmarks/<case>`.

Each benchmark directory contains two files:

- `inputs.json` – simulation configuration passed to `DPFSimulation`.
- `expected.json` – reference time histories and tolerance bands for
  current, voltage and neutron yield.

Running the command produces a pass/fail dashboard and a PNG overlay for
the three diagnostics:

```bash
$ dpf2 run-benchmark unu_pff --benchmark-dir Reference/Benchmarks --output results
```

The generated plot includes grey tolerance bands around the reference
traces with the actual simulation output overlaid.  A table lists whether
current, voltage and neutron yield fall within their respective
bands.
