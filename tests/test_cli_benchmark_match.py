from click.testing import CliRunner

import pydantic_stub
import h5py_stub
import sys
import types
import json
from pathlib import Path

# Stub optional dependencies
sys.modules["pydantic"] = pydantic_stub
sys.modules["pydantic.dataclasses"] = pydantic_stub.dataclasses
sys.modules["h5py"] = h5py_stub


class _DummyAxes:
    def plot(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def axis(self, *args, **kwargs):
        return None

    def text(self, *args, **kwargs):
        return None


class _DummyFig:
    def tight_layout(self):
        return None

    def savefig(self, path):
        Path(path).touch()


class _DummyPyplot:
    def subplots(self, *args, **kwargs):
        return _DummyFig(), _DummyAxes()

    def close(self, *args, **kwargs):
        return None


dummy_pyplot = _DummyPyplot()
sys.modules["matplotlib"] = types.SimpleNamespace(pyplot=dummy_pyplot)
sys.modules["matplotlib.pyplot"] = dummy_pyplot

import importlib

benchmark_cli = importlib.import_module("dpf2.cli.benchmark")
match_benchmark = benchmark_cli.match_benchmark


def test_match_benchmark(tmp_path):
    bench = tmp_path / "bench.csv"
    bench.write_text("time,value\n0,0\n1,1\n")
    sim = tmp_path / "sim.csv"
    sim.write_text("time,value\n0,0\n1,1\n")

    cfg = {
        "dataset_id": "PF1000",
        "benchmark_trace_path": str(bench),
        "benchmark_format": "csv",
        "benchmark_fields": ["I(t)"],
        "waveform_tolerance": 5.0,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    runner = CliRunner()
    result = runner.invoke(
        match_benchmark,
        [
            "--config",
            str(cfg_path),
            "--simulation",
            str(sim),
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report.pdf").exists()
