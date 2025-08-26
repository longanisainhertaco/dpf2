import sys
import types

import click
from click.testing import CliRunner


# Provide a minimal stub for the optional pydantic dependency so the CLI can be
# imported without installing the real package.
pydantic_stub = types.ModuleType("pydantic")
pydantic_dataclasses = types.ModuleType("pydantic.dataclasses")


def _identity_dataclass(cls=None, **kwargs):  # pragma: no cover - trivial stub
    return cls


pydantic_dataclasses.dataclass = _identity_dataclass
pydantic_stub.dataclasses = pydantic_dataclasses
sys.modules.setdefault("pydantic", pydantic_stub)
sys.modules.setdefault("pydantic.dataclasses", pydantic_dataclasses)


import importlib

cli_main = importlib.import_module("dpf2.cli.main")
main = cli_main.main


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

        def run(self, output_dir):
            captured["output"] = output_dir

    monkeypatch.setattr(cli_main.DPFConfig, "from_file", staticmethod(fake_from_file), raising=False)
    monkeypatch.setattr(cli_main, "DPFSimulation", DummySim)

    runner = CliRunner()
    result = runner.invoke(main, ["simulate", "-c", str(config_path), "-o", str(output_dir)])

    assert result.exit_code == 0
    assert captured["config"] == str(config_path)
    assert captured["output"] == str(output_dir)
