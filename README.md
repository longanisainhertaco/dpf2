# DPF2 Simulator

A Dense Plasma Focus (DPF) simulator implemented in Python. This project provides a command line interface and simple models for the external circuit and pinch dynamics.
The purpose of this project is to experiment with LLM capabilities and to use existing, verified research papers to drive a dense plasma focus simulation tool, that will be open
to the scientific community to further explore.

## Documentation

Full documentation, including a user guide, CLI reference, and API reference, lives in the [docs/](docs/index.md) directory and is available as a MkDocs site. Build and view it locally with:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

For details on the command line interface, see [docs/cli.md](docs/cli.md).

## Installation

```bash
pip install .
```

Optional extras can be installed to enable additional, experimental features:

```bash

# Install with Flask-based server support (experimental)
pip install .[server]

# Install with WarpX accelerator support (experimental)
pip install .[warpx]

# Install with OpenCensus-based telemetry and tracing support (experimental)
pip install .[telemetry]

# Install with AMReX/ADIOS2-based radiation support
pip install .[radiation]

# Install with VTK output support
pip install .[diagnostics]

# Install all optional features
pip install .[server,warpx,telemetry,radiation,diagnostics]
```

## Quickstart

Run a simulation using a configuration file:

```bash
dpf2 simulate -c config.json -o results.json
```

Or run a simulation programmatically:

```python
from dpf2 import DPFConfig, DPFSimulation

config = DPFConfig()
simulation = DPFSimulation(config)
result = simulation.run()
```

Configuration files use the `DPFConfig` schema defined in this repository. See the [quickstart tutorial](docs/tutorials/quickstart.md) or its [Jupyter notebook](examples/notebooks/quickstart.ipynb) for a walk-through. For an interactive visualization of sheath dynamics with sliders controlling voltage and pressure, explore [sheath_animation.ipynb](examples/notebooks/sheath_animation.ipynb). Launch the notebooks directly with:

```bash
dpf2 --notebook
```

### Equation of State Backends

The simulator supports both tabulated and ideal–gas equations of state. The
backend is selected via the ``eos_model`` entry in the configuration.  For a
simple ideal gas:

```json
{
  "physics_models": {
    "eos_model": "ideal_gas",
    "gamma": 1.4,
    "mu": 2.0
  }
}
```

A tabulated EOS with a two–species mixture can be configured as:

```json
{
  "physics": {
    "eosModel": "tabulated",
    "eosTablePath": "tests/data/sesame_dummy.csv",
    "mixtureFractions": "Ar:0.9,He:0.1"
  }
}
```

## Server Mode (Experimental)

A lightweight Flask server is provided for remote execution and basic job management. The implementation is not production ready and exposes only minimal endpoints. Install the `[server]` extra and launch with `python -m dpf2.simulation.dpf_simulator_server`. See [`src/dpf2/simulation/dpf_simulator_server.py`](src/dpf2/simulation/dpf_simulator_server.py) and [`tests/test_api_endpoints.py`](tests/test_api_endpoints.py) for the current interface.

## WarpX Accelerator Support (Experimental)

Preliminary integration with [WarpX](https://warpx.readthedocs.io) enables particle-in-cell acceleration via `pywarpx`. The feature is incomplete and intended for experimentation only. Install the `[warpx]` extra and consult [`src/dpf2/warpx_settings.py`](src/dpf2/warpx_settings.py) and [`tests/test_warpx_settings.py`](tests/test_warpx_settings.py) for examples of the configuration schema.

## Telemetry and Tracing (Experimental)

The standalone `dpf_simulation` entry point can emit OpenCensus trace spans around major simulation stages. Enable this optional tracing feature with the `--enable-tracing` flag (requires the `[telemetry]` extra):

```bash
dpf_simulation --config-file config.json --enable-tracing
```

Telemetry streaming through ADIOS2 is available only in certain backends and lacks comprehensive tests. See [`src/dpf2/simulation/dpf_simulator_amrex_backend.py`](src/dpf2/simulation/dpf_simulator_amrex_backend.py) for the experimental implementation and [`src/dpf2/simulation/dpf_simulation.py`](src/dpf2/simulation/dpf_simulation.py) for the tracing hooks. Tracing behaviour is exercised in [`tests/test_dpf_simulation_run.py`](tests/test_dpf_simulation_run.py).

## Repository Layout

- `dpf2/` – simulator package (CLI, circuit solver, plasma model, engine)
- `*config.py` – configuration schemas
- `tests/` – unit and integration tests
- `examples/` – example scripts and notebook

## AI Surrogate Models (Experimental)

The package defines a flexible `SurrogateModel` interface allowing inference with PyTorch or ONNX models, but no pretrained models are included. These classes require the user to install `torch` or `onnxruntime` separately. Surrogates can be plugged into simulations to replace expensive physics modules. See [`src/dpf2/ai/surrogate.py`](src/dpf2/ai/surrogate.py) and [`tests/test_ai_surrogate.py`](tests/test_ai_surrogate.py) for the current interface and coverage.
