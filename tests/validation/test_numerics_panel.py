import json
import pytest

from dpf2.validation.numerics_panel import NumericsPanel
from dpf2.diagnostics.quality_dashboard import QualityDashboard


def test_brio_wu_run(tmp_path):
    q = QualityDashboard(output_dir=tmp_path / "quality")
    panel = NumericsPanel(output_dir=tmp_path / "numerics", quality=q)
    metrics = panel.run_brio_wu()
    out_file = tmp_path / "numerics" / "brio_wu.json"
    assert out_file.exists()
    data = json.load(open(out_file))
    assert "l1_error" in data
    q_data = json.load(open(tmp_path / "quality" / "numerics.json"))
    assert q_data[0]["l1_error"] == metrics["l1_error"]


def test_quality_thresholds(tmp_path, caplog):
    q = QualityDashboard(
        output_dir=tmp_path / "quality",
        max_l1_error=1e-6,
        max_divB_norm=1e-6,
        max_energy_drift=1e-6,
    )
    panel = NumericsPanel(output_dir=tmp_path / "numerics", quality=q)
    panel.run_brio_wu()
    text = caplog.text
    assert "L1 error above threshold" in text
    assert "∇·B norm above threshold" in text
    assert "Energy drift above threshold" in text
