# Run & Compare

The `run-compare` command executes frozen benchmark configurations and
compares the results against reference outputs stored under
`Reference/Benchmarks/`.

Each benchmark directory contains two files:

- `inputs.json` – simulation configuration passed to `DPFSimulation`.
- `expected.json` – reference time histories and tolerance bands for
  current, voltage and neutron yield.

Running the command produces a pass/fail dashboard and PNG overlays for
all three diagnostics:

```bash
$ dpf2 run-compare --benchmark-dir Reference/Benchmarks --output results
```

The generated plots include grey tolerance bands around the reference
traces with the actual simulation output overlaid.  A summary table lists
whether each diagnostic falls within its tolerance band.
