import sys
import types
from pathlib import Path

import h5py_stub

# Provide stubs for optional dependencies
sys.modules["h5py"] = h5py_stub


class _DummyAxes:
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

from scripts import run_benchmark


def test_run_benchmark(tmp_path):
    metrics = run_benchmark.run_benchmark(
        "unu_pff", benchmark_dir="benchmarks", output=str(tmp_path)
    )
    assert metrics["passed"]
    case_dir = tmp_path / "unu_pff"
    assert (case_dir / "overlay.png").exists()
    assert (case_dir / "metrics.json").exists()
    h5_path = case_dir / "results.h5"
    assert h5_path.exists()
    import h5py

    with h5py.File(h5_path, "r") as f:
        assert "manifest" in f
