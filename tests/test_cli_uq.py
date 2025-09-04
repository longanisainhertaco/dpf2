import dataclasses
import json
from pathlib import Path

from click.testing import CliRunner

from dpf2.core.config import DPFConfig
from dpf2.cli import main as cli_main


def test_uq_sweep_and_stats(tmp_path, monkeypatch):
    class DummySim:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            # Peak current proportional to charging voltage for predictability
            return [0.0, 1.0], [self.cfg.charging_voltage * 1e-3], [0.0, 0.0]

    monkeypatch.setattr(cli_main, "DPFSimulation", DummySim)

    cfg = DPFConfig()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(dataclasses.asdict(cfg)))

    runner = CliRunner()
    out_file = tmp_path / "results.json"
    params = '{"charging_voltage":[10000,20000]}'
    res = runner.invoke(
        cli_main.main,
        [
            "uq-sweep",
            "--config",
            str(cfg_path),
            "--parameters",
            params,
            "--samples",
            "2",
            "--method",
            "lhs",
            "--output",
            str(out_file),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert "sobol_indices" in data
    assert "uncertainty_band" in data
    assert len(data["results"]) == 2

    res2 = runner.invoke(cli_main.main, ["uq-stats", "--input", str(out_file)])
    assert res2.exit_code == 0, res2.output
    stats = json.loads(res2.output.strip())
    assert "mean_peak_current" in stats
    assert "std_peak_current" in stats
