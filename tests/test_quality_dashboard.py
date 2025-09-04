import json
import pytest
from dpf2.diagnostics.quality_dashboard import QualityDashboard


def test_quality_dashboard_logs_and_warns(tmp_path, caplog):
    q = QualityDashboard(output_dir=tmp_path, min_cfl=0.5)
    q.log(step=1, dt=0.1, cell_size=1.0, ppc=10, cfl=0.1, lambda_D=1.0)
    data = json.load(open(tmp_path / "dashboard.json"))
    assert data[0]["step"] == 1
    assert "CFL below threshold" in caplog.text


def test_quality_dashboard_abort(tmp_path):
    q = QualityDashboard(output_dir=tmp_path, min_cfl=0.5, abort_on_violation=True)
    with pytest.raises(RuntimeError):
        q.log(step=1, dt=0.1, cell_size=1.0, ppc=10, cfl=0.1, lambda_D=1.0)
