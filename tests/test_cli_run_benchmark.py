from click.testing import CliRunner

import pydantic_stub
import h5py_stub
import sys
import types
from pathlib import Path

# Stub optional dependencies
a = pydantic_stub
sys.modules["pydantic"] = a
sys.modules["pydantic.dataclasses"] = a.dataclasses
sys.modules["h5py"] = h5py_stub
import h5py


class _DummyAxes:
    transAxes = object()

    def plot(self, *args, **kwargs):
        return None

    def fill_between(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
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
    def subplots(self, nrows, ncols, figsize=None):
        fig = _DummyFig()
        axes = [_DummyAxes() for _ in range(nrows)]
        return fig, axes

    def close(self, fig):
        return None


dummy_pyplot = _DummyPyplot()
sys.modules["matplotlib"] = types.SimpleNamespace(pyplot=dummy_pyplot)
sys.modules["matplotlib.pyplot"] = dummy_pyplot


import importlib

cli_main = importlib.import_module("dpf2.cli.main")
main = cli_main.main


def test_run_benchmark_cmd(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "benchmark",
            "run",
            "UNU",
            "--benchmark-dir",
            str(Path("benchmarks")),
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    case_dir = tmp_path / "UNU"
    assert (case_dir / "overlay.png").exists()
    assert (case_dir / "metrics.json").exists()
    h5_path = case_dir / "results.h5"
    assert h5_path.exists()
    with h5py.File(h5_path, "r") as f:
        assert "manifest" in f
    assert "PASSED" in result.output
