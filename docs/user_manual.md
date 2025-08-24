# DPF2 User Manual

## Installation

Install the simulator and its dependencies using pip:

```bash
pip install -e .
```

This will install the `dpf2` command line interface and Python package in editable mode so local changes are immediately available.

## Configuration Schema

Simulations are configured with the `DPFConfig` data class.  It defines the geometry, initial conditions, and numerical parameters for a run.  Configuration files are written in JSON and validated against this schema.  Refer to `dpf_config.py` and related `*config.py` modules for field descriptions and default values.

## CLI Usage

The package installs a `dpf2` entry point that exposes common operations.  Run a simulation with:

```bash
dpf2 simulate config.json -o results.json
```

Use `dpf2 --help` or `dpf2 <command> --help` to see available commands and options.

## Module Extension Points

`dpf2` is designed for extensibility.  New physics models, circuit solvers, or diagnostics can be integrated by subclassing the following base classes:

- `PlasmaSolverBase` – implement alternative plasma evolution models.
- `CircuitSolverBase` – plug in custom external circuit solvers.
- `DiagnosticsBase` – add new diagnostics or analysis routines.
- `SurrogateModel` – wrap AI models that replace expensive physics modules.

Subclass the appropriate base class, register it in your configuration, and the simulation engine will use your implementation.

