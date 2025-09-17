import sys

import click
from click.testing import CliRunner
import json
from pathlib import Path


# Provide a minimal stub for the optional pydantic dependency so the CLI can be
# imported without installing the real package.
import pydantic_stub

sys.modules["pydantic"] = pydantic_stub
sys.modules["pydantic.dataclasses"] = pydantic_stub.dataclasses


import importlib

cli_main = importlib.import_module("dpf2.cli.main")
main = cli_main.main

if not hasattr(cli_main.SyntheticDiagnostics, "parse_obj"):
    cli_main.SyntheticDiagnostics.parse_obj = classmethod(lambda cls, d: cls(**d))


def test_simulate_accepts_short_flags(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "out"

    captured = {}

    def fake_from_file(path):
        captured["config"] = path
        return object()

    class DummySim:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        def run(self, output_dir, verbose=False):
            captured["output"] = output_dir
            captured["verbose"] = verbose
            return [0.0], [0.0], [0.0]

    monkeypatch.setattr(
        cli_main.DPFConfig, "from_file", staticmethod(fake_from_file), raising=False
    )
    monkeypatch.setattr(cli_main, "DPFSimulation", DummySim)

    runner = CliRunner()
    result = runner.invoke(
        main, ["simulate", "-c", str(config_path), "-o", str(output_dir)]
    )

    assert result.exit_code == 0
    assert captured["config"] == str(config_path)
    assert captured["output"] == str(output_dir)


def test_simulate_emits_synthetic(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    synth_cfg = tmp_path / "synthetic.json"

    synth_cfg.write_text(
        json.dumps(cli_main.SyntheticDiagnostics.with_defaults().model_dump())
    )

    class DummySim:
        def __init__(self, cfg):
            pass

        def run(self, output_dir, verbose=False):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return [0.0, 1e-6], [0.0, 1.0], [0.0, 0.0]

    monkeypatch.setattr(cli_main, "DPFSimulation", DummySim)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "simulate",
            "-o",
            str(output_dir),
            "--synthetic",
            str(synth_cfg),
        ],
    )

    assert result.exit_code == 0
    out_file = output_dir / "synthetic_signals.json"
    assert out_file.exists()
