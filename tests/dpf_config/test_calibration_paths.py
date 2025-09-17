import json
from pathlib import Path
from datetime import datetime

from dpf2.dpf_config import DPFConfig


def _serialize(model):
    if isinstance(model, Path):
        return str(model)
    if isinstance(model, datetime):
        return model.isoformat()
    if hasattr(model, "model_dump"):
        try:
            data = model.model_dump()
        except Exception:
            data = model.__dict__
        if isinstance(data, dict):
            return {k: _serialize(v) for k, v in data.items()}
    if isinstance(model, dict):
        return {k: _serialize(v) for k, v in model.items()}
    if isinstance(model, list):
        return [_serialize(v) for v in model]
    return model


def test_calibration_paths_roundtrip(tmp_path):
    cfg = DPFConfig.with_defaults()
    cfg.diagnostics.rogowski_calibration_path = "rog.h5"
    cfg.diagnostics.bdot_calibration_path = "bdot.h5"
    cfg.diagnostics.sxr_calibration_path = "sxr.h5"
    cfg.diagnostics.neutron_tof_calibration_path = "tof.h5"

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_serialize(cfg)))

    loaded = DPFConfig.from_file(path)
    d = loaded.diagnostics
    if isinstance(d, dict):
        assert d["rogowski_calibration_path"] == "rog.h5"
        assert d["bdot_calibration_path"] == "bdot.h5"
        assert d["sxr_calibration_path"] == "sxr.h5"
        assert d["neutron_tof_calibration_path"] == "tof.h5"
    else:
        assert d.rogowski_calibration_path == "rog.h5"
        assert d.bdot_calibration_path == "bdot.h5"
        assert d.sxr_calibration_path == "sxr.h5"
        assert d.neutron_tof_calibration_path == "tof.h5"


def test_calibration_paths_default_none(tmp_path):
    cfg = DPFConfig.with_defaults()
    data = _serialize(cfg)
    diag = data["diagnostics"]
    diag.pop("rogowski_calibration_path", None)
    diag.pop("bdot_calibration_path", None)
    diag.pop("sxr_calibration_path", None)
    diag.pop("neutron_tof_calibration_path", None)

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data))

    loaded = DPFConfig.from_file(path)
    d = loaded.diagnostics
    if isinstance(d, dict):
        assert d.get("rogowski_calibration_path") is None
        assert d.get("bdot_calibration_path") is None
        assert d.get("sxr_calibration_path") is None
        assert d.get("neutron_tof_calibration_path") is None
    else:
        assert d.rogowski_calibration_path is None
        assert d.bdot_calibration_path is None
        assert d.sxr_calibration_path is None
        assert d.neutron_tof_calibration_path is None
