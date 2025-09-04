# Command Line Interface

The `dpf2` executable exposes several subcommands for running
simulations and inspecting results.

## simulate

```
dpf2 simulate -c config.json -o output_dir
```

Run a simulation using a configuration file. Results are written to the
specified output directory.

### Lab mode

Most commands accept the ``--lab-mode`` flag to capture a manifest for
reproducibility. The manifest records the current code hash, RNG seeds,
particle-per-cell setting and any configuration file paths. Both
``manifest.json`` and an accompanying ``manifest.h5`` (with the same
metadata stored as HDF5 attributes) are written inside the command's
output directory.

```
dpf2 simulate -c config.json -o output_dir --lab-mode
```

## wizard

```
dpf2 wizard -o my_config.json
```

Interactive flow that guides users through building a configuration.
Advanced mesh and timing options are available when requested.

## validate

```
dpf2 validate --config config.json --dataset PF1000
```

Execute a simulation and compare the output with bundled validation
traces. Overlay plots are written to the `validation` directory by
default.

## plot

```
dpf2 plot --input output_dir --output plot.png
```

Generate a quick plot of current and voltage from simulation output
files.

## schema

```
dpf2 schema
```

Print a JSON description of the configuration schema and default values.

## latin-hypercube

```
dpf2 latin-hypercube --parameters '{"capacitance":[1e-6,5e-6]}' --samples 8 --output lhs.json
```

Generate Latin hypercube samples for the provided parameter bounds.
The resulting JSON file contains one object per sample.

## sobol-sample

```
dpf2 sobol-sample --parameters '{"capacitance":[1e-6,5e-6]}' --samples 8 --output sobol.json
```

Produce Sobol sequence samples for batch sweeps.

## uq-sweep

```
dpf2 uq-sweep --config config.json --parameters '{"capacitance":[1e-6,5e-6]}' --method lhs --samples 8
```

Execute a sweep across the sampled configurations and record peak current
for each case in ``uq_results.json``.
