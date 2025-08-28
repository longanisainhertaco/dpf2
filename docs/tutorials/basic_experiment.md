# Basic DPF Experiment Tutorial

This tutorial mirrors a simple dense plasma focus experiment. It guides you through
building a configuration, running a simulation, and inspecting results.

## 1. Assemble configuration

1. Copy the [basic simulation template](../config_templates/basic_simulation.yaml).
2. Adjust **device size** and **capacitor bank** parameters to match your setup.
3. Set the desired **fill gas** and pressure.

## 2. Run the wizard

You can also generate a configuration interactively:

```bash
python -m dpf2.cli.main wizard -o my_config.json
```

Answer the prompts for device dimensions, fill gas, and capacitor bank values.

## 3. Start the simulation

```bash
python -m dpf2.cli.main simulate -c my_config.json -o output
```

The solver writes diagnostics to the `output` directory.

## 4. Inspect diagnostics

For a quick look in your browser, launch the web dashboard:

```bash
python -m dpf2.web.app
```

Then open `http://localhost:5000` to run simulations and browse generated files.

## 5. Compare with experiment

Use the diagnostic plots to compare peak current, voltage trace, or neutron yield
with your lab measurements. Iteratively adjust the configuration until the
simulated behavior matches the experiment.
