# Advanced Physics Tutorial

Explore high-fidelity plasma models and optional physics packages.

## 1. Enable extended models

Activate the Hall MHD solver and radiation transport by installing the
appropriate extras:

```bash
pip install .[radiation]
```

Update your configuration to switch on the modules:

```yaml
physics:
  hall_mhd: true
  radiation: true
```

## 2. Run with detailed diagnostics

The additional physics can slow the solver, so enable only the
instrumentation you need:

```bash
dpf2 simulate -c advanced.yml -o run --diagnostics
```

Current and voltage traces now include the effects of the activated
models. Inspect the history file or use the plotting helpers to compare
runs.

## 3. Calibrate model parameters

Experimental campaigns often require tuning uncertain parameters to fit
diagnostics.  The :mod:`dpf2.uq.inference` module provides wrappers
around modern samplers such as ``emcee`` and ``dynesty``:

```python
from dpf2.uq.inference import emcee_infer

def model(params):
    return run_simulation(params)  # user-defined helper

samples = emcee_infer(model, {"charging_voltage": (8e3, 12e3)}, data)
```

The returned dictionary maps each parameter to an array of posterior
samples that can be analysed or passed to downstream studies.

## 4. Quantify sensitivity and uncertainty

The ``uq-sweep`` command now reports Sobol indices and a 95% uncertainty
band for the output metric:

```bash
dpf2 uq-sweep --config cfg.json --parameters '{"charging_voltage":[1e4,2e4]}' \
    --samples 16 --method sobol
```

The generated JSON file contains per-run results along with a
``sobol_indices`` block and an ``uncertainty_band`` describing the mean
response and confidence interval.
