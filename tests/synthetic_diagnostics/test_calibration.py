import json
import pytest

from dpf2.synthetic_diagnostics import (
    SyntheticDiagnostics,
    SyntheticInstrument,
    run_diagnostic_calculations,
)
from dpf2.core.bases import CouplingState


def _history():
    return [
        CouplingState(current=1.0, voltage=0.0),
        CouplingState(current=2.0, voltage=0.0),
    ]


def test_rogowski_calibration(tmp_path):
    hist = _history()
    cal = tmp_path / "rogowski.json"
    cal.write_text(json.dumps({"scale": 2.0}))
    cfg = SyntheticDiagnostics.with_defaults().model_copy(
        update={
            "apply_time_response": True,
            "instrument_response_directory": tmp_path,
            "synthetic_rogowski_signal_enabled": True,
            "instrument_overrides": {"rogowski": SyntheticInstrument(calibration_file=cal)},
        }
    )
    out = run_diagnostic_calculations(hist, cfg, dt=1.0)
    cfg_base = SyntheticDiagnostics.with_defaults().model_copy(
        update={"synthetic_rogowski_signal_enabled": True}
    )
    out_base = run_diagnostic_calculations(hist, cfg_base, dt=1.0)
    assert out_base["rogowski"] != out["rogowski"]
    expected = [v * 2.0 for v in out_base["rogowski"]]
    assert all(abs(a - b) < 1e-8 for a, b in zip(out["rogowski"], expected))


def test_bdot_calibration(tmp_path):
    hist = _history()
    cal = tmp_path / "bdot.json"
    cal.write_text(json.dumps({"scale": 0.5}))
    cfg = SyntheticDiagnostics.with_defaults().model_copy(
        update={
            "apply_time_response": True,
            "instrument_response_directory": tmp_path,
            "synthetic_bdot_signal_enabled": True,
            "instrument_overrides": {"bdot": SyntheticInstrument(calibration_file=cal)},
        }
    )
    out = run_diagnostic_calculations(hist, cfg, dt=1.0)
    cfg_base = SyntheticDiagnostics.with_defaults().model_copy(
        update={"synthetic_bdot_signal_enabled": True}
    )
    out_base = run_diagnostic_calculations(hist, cfg_base, dt=1.0)
    assert out_base["bdot"] != out["bdot"]
    expected = [v * 0.5 for v in out_base["bdot"]]
    assert all(abs(a - b) < 1e-12 for a, b in zip(out["bdot"], expected))


def test_constant_calibration_applies_full_gain():
    from dpf2.diagnostics.synthetic_signals import _apply_instrument_response

    values = [1.0, 2.0, 3.0, 4.0]
    dt = 1.0
    scale = 2.0
    result = _apply_instrument_response(values, dt, [0.0], [scale])
    assert result == [v * scale for v in values]

