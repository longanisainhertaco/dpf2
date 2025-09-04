import sys
import json
from pathlib import Path

from click.testing import CliRunner

import pydantic_stub
import h5py_stub

# Stub optional dependencies
sys.modules["pydantic"] = pydantic_stub
sys.modules["pydantic.dataclasses"] = pydantic_stub.dataclasses
sys.modules["h5py"] = h5py_stub

import importlib
import types


class _DummyPyplot:
    def figure(self):
        return None

    def plot(self, *args, **kwargs):
        return None

    def xlabel(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def tight_layout(self, *args, **kwargs):
        return None

    def savefig(self, path):
        Path(path).touch()


_dummy_pyplot = _DummyPyplot()
sys.modules["matplotlib"] = types.SimpleNamespace(pyplot=_dummy_pyplot)
sys.modules["matplotlib.pyplot"] = _dummy_pyplot

cli_main = importlib.import_module("dpf2.cli.main")
main = cli_main.main


def test_validate_config_cmd(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({}))
    runner = CliRunner()
    result = runner.invoke(main, ["validate-config", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "Configuration is valid" in result.output


def test_validate_config_cmd_invalid(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({}))

    def fake_from_file(path):
        raise cli_main.ConfigurationError("bad", ["charging_voltage"])

    monkeypatch.setattr(
        cli_main.DPFConfig, "from_file", staticmethod(fake_from_file), raising=False
    )

    runner = CliRunner()
    result = runner.invoke(main, ["validate-config", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "charging_voltage" in result.output


def test_plot_run_cmd(tmp_path):
    fname = tmp_path / "data_0000.h5"
    with h5py_stub.File(fname, "w") as fh:
        fh.create_dataset("time", data=0.0)
        fh.create_dataset("current", data=1.0)
        fh.create_dataset("voltage", data=2.0)
    runner = CliRunner()
    out_png = tmp_path / "plot.png"
    result = runner.invoke(main, [
        "plot-run",
        "--run-dir",
        str(tmp_path),
        "--output",
        str(out_png),
    ])
    assert result.exit_code == 0
    assert out_png.exists()


def test_student_flag_invokes_gui(monkeypatch):
    called: dict[str, bool] = {}

    def fake_launch(*, simplified: bool = False):
        called["simplified"] = simplified

    monkeypatch.setattr(cli_main, "interactive", types.SimpleNamespace(launch=fake_launch))
    runner = CliRunner()
    result = runner.invoke(main, ["--student"])
    assert result.exit_code == 0
    assert called.get("simplified") is True
