# Parameter Sweep Workflow

Engineers often explore design space through automated parameter sweeps.

## Procedure

1. Choose parameters to vary such as voltage, pressure, or anode radius.
2. Generate sweep configurations using the provided scripts.
3. Launch the sweep runner to execute multiple simulations.
4. Aggregate results and identify trends to guide design choices.

Sweeps accelerate design iteration and sensitivity analysis.

## Training Surrogate Models

Generate lightweight surrogates for yield and pinch time using the
benchmark datasets:

```bash
python scripts/train_surrogates.py
ls ai/surrogates
```

Expected output:

```
pinch_time_model.json  yield_model.json  yield_model.onnx
```

## Metadata and OOD Checks

Each surrogate records its training domain, error and optional
distribution statistics. Predictions outside this range trigger an
`OptimizationWarning`:

```bash
PYTHONPATH=src python - <<'PY'
from dpf2.ai import load_yield_surrogate
import warnings
model = load_yield_surrogate()
print("domain:", model.domain)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model.predict(model.domain[1] + 10)
    print("warning:", w[0].message)
PY
```

Example output:

```
domain: [90.0, 200.0]
warning: Input 210.0 outside training range [90.0, 200.0] (distance=0.00)
```

## Parameter Sweeps with Surrogates

With surrogates in place, parameter sweeps can quickly estimate KPIs
without running full simulations:

```bash
dpf2 param-sweep --config examples/config.json \
    --parameter initial_pressure --values 5 10 15 \
    --output sweep_surrogate --kpi
```

Expected output:

```
Sweep complete. Results written to sweep_surrogate
```
