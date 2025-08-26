import numpy as np
import pytest

pyevtk = pytest.importorskip("pyevtk")

from dpf2.simulation.diagnostics import Diagnostics


def test_to_vtk_writes_file(tmp_path):
    diag = Diagnostics(
        str(tmp_path / "out.h5"),
        {},
        (0.0, 0.0, 0.0),
        (1, 1, 1),
        1.0,
        5.0 / 3.0,
        object(),
    )
    snap = {
        "time": 0.0,
        "density": np.zeros((1, 1, 1)),
        "pressure": np.zeros((1, 1, 1)),
        "velocity": np.zeros((1, 1, 1, 3)),
        "magnetic": np.zeros((1, 1, 1, 3)),
    }
    diag.snapshots.append({"snapshot": snap, "checkpoint": None})
    diag.to_vtk(str(tmp_path / "snap"))
    assert (tmp_path / "snap_0.vtr").exists()
