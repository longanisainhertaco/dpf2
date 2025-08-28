import json
from click.testing import CliRunner

from dpf2.cli.main import diagnostics


def test_cli_outputs_current_and_voltage(tmp_path):
    history = [
        {"Lp": 0.0, "emf": 0.0, "current": 0.0, "voltage": 1.0},
        {"Lp": 0.0, "emf": 0.0, "current": 1.0, "voltage": 0.5},
        {"Lp": 0.0, "emf": 0.0, "current": 2.0, "voltage": 0.0},
    ]
    hist_path = tmp_path / "history.json"
    hist_path.write_text(json.dumps(history))

    runner = CliRunner()
    result = runner.invoke(
        diagnostics,
        ["--history", str(hist_path), "--current", "--voltage"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["current"] == [0.0, 1.0, 2.0]
    assert data["voltage"] == [1.0, 0.5, 0.0]

