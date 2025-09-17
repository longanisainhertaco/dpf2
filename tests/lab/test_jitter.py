import json
from statistics import mean, stdev
from click.testing import CliRunner
import h5py_stub  # register stubbed h5py

from dpf2.core.config import DPFConfig, JitterConfig, JitterDistribution
from dpf2.cli.lab import lab


def test_jitter_statistics_and_manifest(tmp_path):
    cfg = DPFConfig(
        end_time=1e-7,
        jitter=JitterConfig(
            voltage=JitterDistribution(std=0.05),
            pressure=JitterDistribution(std=0.10),
            switch_timing=JitterDistribution(std=5.0),
        ),
    )
    cfg_path = tmp_path / "cfg.json"
    cfg.to_file(cfg_path)

    runner = CliRunner()
    shots = 20
    result = runner.invoke(
        lab,
        ["run", "--config", str(cfg_path), "--shots", str(shots), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output

    manifest = tmp_path / "out" / "batch_manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    volts = [r["voltage"] for r in data["runs"]]
    press = [r["pressure"] for r in data["runs"]]
    switch = [r["switch_timing"] for r in data["runs"]]

    assert abs(mean(volts) - cfg.charging_voltage) < cfg.charging_voltage * 0.1
    assert abs(stdev(volts) - cfg.charging_voltage * 0.05) < cfg.charging_voltage * 0.03
    assert abs(mean(press) - cfg.initial_pressure) < cfg.initial_pressure * 0.1
    assert abs(stdev(press) - cfg.initial_pressure * 0.10) < cfg.initial_pressure * 0.05
    assert abs(mean(switch)) < 5e-9
    assert abs(stdev(switch) - 5e-9) < 3e-9
    # ensure seeds captured
    assert all("seeds" in r for r in data["runs"])
