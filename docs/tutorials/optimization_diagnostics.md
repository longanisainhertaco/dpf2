# Optimization and Diagnostics Tutorial

Engineers often need to tune device parameters and inspect system health. This
tutorial demonstrates how to automate optimization and gather diagnostics.

## 1. Configure optimization targets

Start with a base configuration and specify the objective in the optimizer
section:

```yaml
optimization:
  target: neutron_yield
  method: bayesian
```

## 2. Launch the optimization loop

```bash
python -m dpf2.cli.main optimize -c my_config.json -o runs/
```

Each iteration stores parameters and results in the `runs/` directory.

## 3. Gather diagnostics

After optimization, collect detailed diagnostics:

```bash
python -m dpf2.cli.main diagnostics -i runs/best -o analysis/
```

Plots of current traces and density profiles appear in `analysis/` to help
identify performance bottlenecks.

## 4. Iterate

Adjust hardware constraints or algorithm settings and rerun the optimizer until
requirements are satisfied.
