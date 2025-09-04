# Run & Compare

The `run-compare` command executes frozen benchmark configurations and
compares the results against reference outputs stored under
`benchmarks/`.

Each benchmark directory contains two files:

- `inputs.json` – simulation configuration passed to `DPFSimulation`.
- `expected.json` – reference time histories and tolerance bands for
  current, voltage and neutron yield.

Running the command produces a pass/fail dashboard and PNG overlays for
all three diagnostics. By default plots are written to `Validation/`:

```bash
$ dpf2 run-compare
```

The generated plots include grey tolerance bands around the reference
traces with the actual simulation output overlaid.  A summary table lists
whether each diagnostic falls within its tolerance band.
