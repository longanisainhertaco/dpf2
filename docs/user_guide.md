# User Guide

## Installation

Install the simulator and its dependencies using pip:

```bash
pip install -e .
```

This installs the `dpf2` command line interface and Python package in editable mode so local changes are immediately available.

Install optional radiation dependencies (AMReX and ADIOS2) with:

```bash
pip install -e .[radiation]
```

Enable VTK output diagnostics with:

```bash
pip install -e .[diagnostics]
```

## Configuration Schema

Simulations are configured with the `DPFConfig` data class. Configuration files are written in JSON and validated against this schema. See `dpf_config.py` and related `*config.py` modules for field descriptions and default values.

### Distributed Circuit Configuration

The circuit section now supports a distributed representation composed of
segments and switches.  Each `SegmentConfig` connects two nodes using the
`from_node` and `to_node` fields and may include fixed parasitic
inductance, resistance and capacitance values as well as optional
time-dependent profiles for these quantities
(`l_profile`, `r_profile`, `c_profile`).  `SwitchConfig` definitions have
gained a `trigger_times` list in nanoseconds along with corresponding
parasitic `l_parasitic`, `r_parasitic` and `c_parasitic` parameters.  The
`CircuitConfig.build_distributed_model()` helper translates these
configurations into `TransmissionLineSegment` and `Switch` instances from
`dpf2.circuit.distributed` for use by network solvers.

## Physics Modules

The core simulation is composed of modular physics components. Key modules include:

- `neutron_yield_model` – estimates fusion yield from plasma conditions.
- `radiation_transport` – models X-ray emission and transport.
- `rlc_solver` – couples the plasma to an external circuit model.

These and other modules can be extended by subclassing their respective base classes.

## Examples

Run the CLI against the provided sample configuration:

```bash
dpf2 simulate -c examples/config.json -o results.json
```

For an interactive walk-through, open the Jupyter notebook in the `examples/notebooks/quickstart.ipynb` file.

## Uncertainty Quantification

The simulator includes basic tools for exploring how input parameters
affect key outputs.  Two sampling schemes are available: Latin hypercube
(`lhs`) and Sobol sequences (`sobol`).  A sweep generates multiple
configurations and records the peak current from each run:

```bash
dpf2 uq-sweep --config config.json \
    --parameters '{"charging_voltage":[14000,16000]}' \
    --method sobol --samples 8 --output sweep.json
```

Summary statistics can then be computed over the sweep results:

```bash
dpf2 uq-stats --input sweep.json
```

The `uq-stats` command prints the mean and standard deviation of the
peak current across all simulated samples.

## Server Usage

A lightweight Flask server is provided for remote execution. Install server extras and launch:

```bash
pip install -e .[server]
python -m dpf2.server
```

Submit configuration JSON to the `/simulate` endpoint to run simulations remotely.
