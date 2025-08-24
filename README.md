# DPF2 Simulator

A minimal Dense Plasma Focus (DPF) simulator implemented in Python. This project provides a command line interface and simple models for the external circuit and pinch dynamics.

## Documentation

Full documentation, including a user guide and API reference, is available as a MkDocs site. Build and view it locally with:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## Installation

```bash
pip install -e .
```

Optional extras can be installed to enable additional features:

```bash
# Install with Flask-based server support
pip install -e .[server]

# Install with WarpX accelerator support
pip install -e .[warpx]

# Install all optional features
pip install -e .[server,warpx]
```

## Quickstart

Run a simulation using a configuration file:

```bash
dpf2 simulate config.json -o results.json
```

Or run a simulation programmatically:

```python
from dpf2 import DPFConfig, DPFSimulation

config = DPFConfig()
simulation = DPFSimulation(config)
result = simulation.run()
```

Configuration files use the `DPFConfig` schema defined in this repository. See `examples/quickstart.ipynb` for a walk-through in a Jupyter notebook.

## Tracing

The standalone `dpf_simulation` entry point can emit OpenCensus trace spans around major simulation stages. Enable this optional feature with the `--enable-tracing` flag (requires the `opencensus` package):

```bash
dpf_simulation --config-file config.json --enable-tracing
```

When tracing is disabled, the simulation runs without importing the tracing library.

## Repository Layout

- `dpf2/` – simulator package (CLI, circuit solver, plasma model, engine)
- `*config.py` – configuration schemas
- `tests/` – unit and integration tests
- `examples/` – example scripts and notebook

## AI Surrogate Models

The package defines a flexible `SurrogateModel` interface allowing inference with PyTorch or ONNX models. Surrogates can be plugged into simulations to replace expensive physics modules. See the `dpf2.ai` module for details.
