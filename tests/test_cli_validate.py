from pathlib import Path

import pydantic

if not hasattr(pydantic.BaseModel, "parse_obj"):  # pragma: no cover - compatibility
    pydantic.BaseModel.parse_obj = classmethod(lambda cls, d: cls(**d))
if not hasattr(pydantic.BaseModel, "model_validate"):  # pragma: no cover - compatibility
    pydantic.BaseModel.model_validate = classmethod(lambda cls, d, **_: cls.parse_obj(d))

from dpf2.cli.validate import run_validation
from dpf2.dpf_config import DPFConfig


def test_run_validation_creates_report(tmp_path):
    cfg = DPFConfig.with_defaults()
    cfg = cfg.model_copy(update={
        "simulation_control": cfg.simulation_control.model_copy(update={"time_end": 1e-7})
    })
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(cfg.model_dump_json())
    outdir = tmp_path / "out"
    ok = run_validation(cfg_path, "PF1000", outdir=outdir)
    assert (outdir / "scaling_report.json").exists()
    assert isinstance(ok, bool)
