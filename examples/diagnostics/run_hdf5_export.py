"""Example of running synthetic diagnostics and exporting to HDF5."""
from pathlib import Path

from dpf2.synthetic_diagnostics import (
    SyntheticDiagnostics,
    run_diagnostic_calculations,
    export_diagnostic_data,
)
from dpf2.core.bases import CouplingState

history = [CouplingState(current=i, voltage=i * 2.0) for i in range(5)]

cfg = SyntheticDiagnostics.with_defaults().model_copy(
    update={
        "synthetic_cr39_image_enabled": True,
        "synthetic_rcf_image_enabled": True,
        "synthetic_faraday_iedf_enabled": True,
        "synthetic_faraday_eedf_enabled": True,
        "diagnostic_output_type": {"cr39_image": "image", "rcf_image": "image"},
    }
)

results = run_diagnostic_calculations(history, cfg, dt=1.0)
export_diagnostic_data(
    results, cfg.model_copy(update={"output_format": "hdf5"}), Path("output_hdf5")
)

print("HDF5 files:", [p.name for p in Path("output_hdf5").glob("*")])
