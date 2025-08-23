# Validation Suite

This project includes a light-weight validation routine in `validation_suite.py` for
comparing simulation results with benchmark data.

The benchmark datasets live in the `Validation/` directory:

- `current_profile.csv` – measured current profile I(t) in kA.
- `inductance_profile.csv` – measured inductance profile L(t) in nH.
- `gv_timing.json` – reference gas-valve (GV) trigger time in microseconds.

Use `compute_error_metrics` to compare simulated outputs against these
benchmarks. The function returns root-mean-square errors for the I(t) and
L(t) profiles, an absolute difference for GV timing, and a `passed` flag
showing whether all metrics fall within user supplied tolerances.

```python
from pathlib import Path
import numpy as np
from validation_suite import compute_error_metrics

sim_outputs = {
    "gv_time_us": 2.6,
    "I": (np.array([0,1,2,3,4,5]), np.array([0,11,19,31,39,52])),
    "L": (np.array([0,1,2,3,4,5]), np.array([10,10.5,12.5,13,14.5,15.5])),
}

metrics = compute_error_metrics(
    sim_outputs, Path("Validation"),
    {"gv_timing_us": 0.2, "I(t)": 5.0, "L(t)": 2.0}
)
print(metrics["passed"])
```
