import h5py_stub as h5py
import numpy as np
from dpf2.simulation.openpmd_io import OpenPMDWriter


def test_openpmd_writer(tmp_path):
    path = tmp_path / "out.h5"
    writer = OpenPMDWriter(path)
    fields = {"E": np.zeros((3, 2, 2, 2)), "B": np.ones((3, 2, 2, 2))}
    writer.write_fields(0, fields)
    particles = {"e": {"x": np.array([0.0, 1.0])}}
    writer.write_particles(0, particles)
    writer.close()
    with h5py.File(path, "r") as f:
        assert f.attrs["openPMD"] == "1.1.0"
        assert "data/0/E" in f
        assert "particles" in f["data/0"]
