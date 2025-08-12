# DPF2 Simulator

A minimal Dense Plasma Focus (DPF) simulator implemented in Python.  This
project provides a command line interface and simple models for the
external circuit and pinch dynamics.

## Installation

```bash
pip install -e .
```

## Quickstart

Run a simulation using the default configuration:

```bash
dpf2 simulate config.json -o results.json
```

Configuration files use the `DPFConfig` schema defined in this repository.
See `examples/quickstart.ipynb` for a walk-through in a Jupyter notebook.

## Repository Layout

- `dpf2/` – simulator package (CLI, circuit solver, plasma model, engine)
- `*config.py` – configuration schemas
- `tests/` – unit and integration tests
- `examples/` – example scripts and notebook

## AI Surrogate Models

The package defines a flexible `SurrogateModel` interface allowing
inference with PyTorch or ONNX models. Surrogates can be plugged into
simulations to replace expensive physics modules. See the `dpf2.ai`
module for details.

## Development

Run the unit and integration tests with:

```bash
pytest
```

Contributions are welcome.  For ideas on how the code could evolve into a high-
performance multi-physics tool see `docs/HPC_DESIGN.md`.
