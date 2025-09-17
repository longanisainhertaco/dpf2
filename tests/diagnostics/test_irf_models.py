import json

import h5py_stub as h5py

from dpf2.diagnostics.detector_models import apply_irf
from dpf2.diagnostics.neutron.tof_synthetic import synthetic_tof_from_iv
from dpf2.diagnostics.xray import apply_response
from dpf2.diagnostics.interferometry import interferometer_phase_shift
from dpf2.synthetic_diagnostics.core import SyntheticDiagnostics, export_diagnostic_data
from types import SimpleNamespace


def test_irf_application_and_metadata(tmp_path):
    irf = {
        "transfer_function": [0.5, 0.5],
        "gating": {"start": 0.0, "end": 3.0},
        "dead_time": 0.5,
        "noise": {"stddev": 0.0},
    }
    times = [0.0, 1.0, 2.0, 3.0]
    signal = [1.0, 1.0, 1.0, 1.0]
    processed = apply_irf(times, signal, irf)

    t0, s0 = synthetic_tof_from_iv([1.0, 1.0], [1.0, 1.0], 1.0, 1.0, [1.0])
    _, s1 = synthetic_tof_from_iv([1.0, 1.0], [1.0, 1.0], 1.0, 1.0, [1.0], irf=irf)
    assert s1 == apply_irf(t0, s0, irf)

    xr = apply_response(times, signal, irf)
    assert xr == processed

    phase0 = interferometer_phase_shift([1.0], [1.0], 1e-6)
    phase1 = interferometer_phase_shift([1.0], [1.0], 1e-6, irf=irf)
    assert phase1 == apply_irf([0.0], [phase0], irf)[0]

    cfg = SimpleNamespace(output_format="hdf5", instrument_response=irf, diagnostic_output_type={})
    paths = export_diagnostic_data({"sig": processed}, cfg, tmp_path)
    with h5py.File(paths[0], "r") as fh:
        ds = fh["sig"]
        assert json.loads(ds.attrs["instrument_response"]) == irf
