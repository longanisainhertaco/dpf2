import json
import pytest
from dpf2.diagnostics.thresholds import ThresholdDashboard


def test_threshold_dashboard_logs_and_warns(tmp_path, caplog):
    dash = ThresholdDashboard(
        output_dir=tmp_path,
        max_cfl=0.5,
        min_lambda_D_dx=2.0,
        max_divB=0.1,
    )
    statuses = dash.log(
        step=1,
        cfl=1.0,
        lambda_D=0.5,
        cell_size=1.0,
        divB=0.2,
    )
    data = json.load(open(tmp_path / "dashboard.json"))
    assert data[0]["step"] == 1
    assert statuses["cfl"] == "red"
    assert statuses["lambda_D_dx"] == "red"
    assert statuses["divB"] == "red"
    assert "CFL above threshold" in caplog.text
    assert "lambda_D/dx below threshold" in caplog.text
    assert "divB above threshold" in caplog.text


def test_threshold_dashboard_abort(tmp_path):
    dash = ThresholdDashboard(output_dir=tmp_path, max_cfl=0.5, abort_on_violation=True)
    with pytest.raises(RuntimeError):
        dash.log(step=1, cfl=1.0, lambda_D=1.0, cell_size=1.0, divB=0.0)
