# Optimization Tutorial

This guide walks engineers through tuning dense plasma focus (DPF) parameters to meet design goals.

## 1. Establish objectives and bounds

Define variables to tune and constraints in a configuration file:

```yaml
optimization:
  target: peak_current
  bounds:
    anode_radius: [0.5e-2, 1.5e-2]
    bank_voltage: [20e3, 40e3]
```

The solver treats bounds as hard limits during optimization.

## 2. Launch a parameter sweep

Evaluate a grid of configurations with the sweep utility:

```bash
python -m dpf2.cli.main sweep -c base.json -p sweep.yaml -o sweep_runs/
```

Each run is tagged with the varied parameters for later sorting.

## 3. Analyze the Pareto front

Collect metrics and plot trade-off surfaces:

```bash
python -m dpf2.cli.main analyze -i sweep_runs/ --pareto neutron_yield energy_input
```

The plot highlights configurations that maximize neutron yield for a given energy input.

## 4. Export the best design

Write the optimal settings to a new configuration:

```bash
python -m dpf2.cli.main best -i sweep_runs/ --metric neutron_yield -o optimal.json
```

Use this file directly in subsequent simulations or share it with collaborators.
