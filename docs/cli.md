# Command Line Interface

The `dpf2` executable exposes several subcommands for running
simulations and inspecting results.

## simulate

```
dpf2 simulate -c config.json -o output_dir
```

Run a simulation using a configuration file. Results are written to the
specified output directory.

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
