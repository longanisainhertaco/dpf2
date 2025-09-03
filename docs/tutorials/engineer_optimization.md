# Engineering Optimization Tutorial

Engineers often refine dense plasma focus hardware by balancing performance
targets with practical limits. This tutorial shows how to run structured design
studies and automatically search parameter space.

## 1. Establish objectives and bounds

Define the variables to tune and any constraints in your configuration file:

```yaml
optimization:
  target: peak_current
  bounds:
    anode_radius: [0.5e-2, 1.5e-2]
    bank_voltage: [20e3, 40e3]
```

The solver treats bounds as hard limits during optimization.

## 2. Launch a parameter sweep

Use the built-in sweep utility to evaluate a grid of configurations:

```bash
python -m dpf2.cli.main sweep -c base.json -p sweep.yaml -o sweep_runs/
```

Each run is tagged with the varying parameters so results can be sorted later.

## 3. Analyze the Pareto front

After the sweep, collect metrics and plot the trade-off surface:

```bash
python -m dpf2.cli.main analyze -i sweep_runs/ --pareto neutron_yield energy_input
```

The plot highlights configurations that maximize neutron yield for a given
energy input.

## 4. Export the best design

Select the optimal run and write its settings to a new configuration:

```bash
python -m dpf2.cli.main best -i sweep_runs/ --metric neutron_yield -o optimal.json
```

This file can be used directly in subsequent simulations or shared with
collaborators.

