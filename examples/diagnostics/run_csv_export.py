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

# Enable additional diagnostics including CR-39/RCF images and Faraday cup
cfg = SyntheticDiagnostics.with_defaults().model_copy(
    update={
        "synthetic_cr39_image_enabled": True,
        "synthetic_rcf_image_enabled": True,
        "synthetic_faraday_iedf_enabled": True,
        "synthetic_faraday_eedf_enabled": True,
        "diagnostic_output_type": {"cr39_image": "image", "rcf_image": "image"},
    }
)

# Compute diagnostic signals and export them to CSV files in ``output_csv``
results = run_diagnostic_calculations(history, cfg, dt=1.0)
export_diagnostic_data(
    results, cfg.model_copy(update={"output_format": "csv"}), Path("output_csv")
)

print("Wrote:", [p.name for p in Path("output_csv").glob("*")])
