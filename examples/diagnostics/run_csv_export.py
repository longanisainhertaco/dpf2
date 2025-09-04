"""Example of running synthetic diagnostics and exporting to CSV."""
from pathlib import Path

from dpf2.synthetic_diagnostics import (
    SyntheticDiagnostics,
    run_diagnostic_calculations,
    export_diagnostic_data,
)
from dpf2.core.bases import CouplingState

# Generate a small history of coupling states
history = [CouplingState(current=i, voltage=i * 2.0) for i in range(5)]

# Use default diagnostic settings which enable current and voltage waveforms
cfg = SyntheticDiagnostics.with_defaults()

# Compute diagnostic signals and export them to CSV files in ``output_csv``
results = run_diagnostic_calculations(history, cfg, dt=1.0)
export_diagnostic_data(results, cfg.model_copy(update={"output_format": "csv"}), Path("output_csv"))

print("Wrote:", [p.name for p in Path("output_csv").glob("*")])
