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
particle-per-cell setting, environment details and any configuration file paths. Both
``run_manifest.json`` and an accompanying ``run_manifest.h5`` (with the same
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

## project sweep

```
dpf2 project sweep --config config.json --grid '{"initial_pressure":[0.08,0.1,0.12]}' --output campaign_runs --manifest
```

Orchestrate a batch of parametric sweeps and store a manifest per sweep point
plus a combined `campaign_metrics.json`. The `--grid` flag accepts an inline JSON
object or a path to a JSON file mapping parameter names to arrays of values. All
runs inherit lab-mode manifest logging when the top-level `--lab-mode` flag is
present.

## project compare

```
dpf2 project compare --parameter initial_pressure --summary baseline=campaign_runs/initial_pressure/summary.json --summary new_mesh=campaign_runs/new_mesh/summary.json
```

Overlay multiple sweep summaries and emit a comparison plot along with a CSV
table. Summary files are the `summary.json` outputs produced by parametric
sweeps or the `campaign_metrics.json` bundle written by `project sweep`.

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

## HPC Workflow Example

The CLI can be combined with batch systems for parallel runs. For example, a four-rank job under SLURM can be launched with:

```bash
srun -n 4 dpf2 simulate -c config.json -o output --lab-mode
```

This executes the solver across multiple tasks while capturing a manifest for reproducibility. Performance characteristics on a reference cluster are shown below.

![Strong scaling](images/strong_scaling.png)
