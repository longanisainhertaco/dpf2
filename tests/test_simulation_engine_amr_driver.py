import sys

import pydantic_stub

sys.modules.setdefault("pydantic", pydantic_stub)
sys.modules.setdefault("pydantic.dataclasses", pydantic_stub.dataclasses)

from dpf2.dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine
import numpy as np


def test_simulation_engine_invokes_amr(monkeypatch):
    cfg = DPFConfig.with_defaults()
    cfg.simulation_control.time_end = 1e-9
    cfg.simulation_control.min_dt = 1e-9
    cfg.parallel_settings.amr_refinement_criteria = {"gradient_threshold": 0.1}
    called = {}

    def fake_refine(self, plasma_state=None, prev_field=None):
        called["refine"] = True

    def fake_run(self, time, current):
        from types import SimpleNamespace

        n = len(time)
        zeros = np.zeros(n)
        return SimpleNamespace(
            time=time,
            radius=zeros,
            temperature=zeros,
            pressure=zeros,
            neutron_yield=0.0,
            axial_position=None,
        )

    monkeypatch.setattr("dpf2.mesh.amr.AMRMesh.refine", fake_refine)
    monkeypatch.setattr("dpf2.pinch_models.AnalyticPinchModel.run", fake_run)

    engine = SimulationEngine(cfg)
    engine.run()
    assert called.get("refine")
