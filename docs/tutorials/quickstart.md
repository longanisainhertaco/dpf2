# Quickstart Tutorial

This tutorial walks through installing DPF2, running a minimal simulation, and inspecting the results. It mirrors the
[accompanying Jupyter notebook](../../examples/notebooks/quickstart.ipynb).

## 1. Install DPF2

Clone the repository and install the package:

```bash
git clone https://github.com/your-org/dpf2.git
cd dpf2
pip install -r requirements.txt
pip install .
```

## 2. Run a simulation

Create a default configuration and execute the solver in Python:

```python
from dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine

cfg = DPFConfig.with_defaults()
engine = SimulationEngine(cfg)
results = engine.run()
```

Alternatively, run the command line interface with an explicit configuration file:

```bash
dpf2 simulate -c examples/config.json -o results.json
```

## 3. Inspect results

Plot the current trace from the Python session:

```python
import matplotlib.pyplot as plt

plt.plot(results.time * 1e6, results.current / 1e3)
plt.xlabel("Time [µs]")
plt.ylabel("Current [kA]")
```

You now have a complete end-to-end workflow: install, run, and analyze. Explore more experiments in the other
[tutorials](./) and notebooks in the [examples/notebooks](../../examples/notebooks/) directory.
