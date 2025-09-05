import json
from pathlib import Path

from dpf2.diagnostics.synthetic_signals import rogowski_signal, angular_neutron_spectrum
from dpf2.core.bases import CouplingState
from dpf2.dpf_config import DPFConfig


def _history():
    return [
        CouplingState(current=1.0, voltage=0.0),
        CouplingState(current=2.0, voltage=0.0),
    ]


def test_rogowski_calibration_from_config(tmp_path):
    cal = tmp_path / "rog.json"
    cal.write_text(json.dumps({"scale": 2.0}))
    cfg = DPFConfig.with_defaults()
    cfg.diagnostics.rogowski_calibration_path = str(cal)
    hist = _history()
    out = rogowski_signal(hist, 1.0, cfg=cfg)
    base = rogowski_signal(hist, 1.0)
    assert out != base
    assert all(abs(o - b * 2.0) < 1e-8 for o, b in zip(out, base))


def test_angular_neutron_spectrum():
    spec = angular_neutron_spectrum([0.0, 90.0, 180.0], 1.0, anisotropy=0.5)
    assert spec[0] > spec[1]
    assert spec[2] < spec[1]
