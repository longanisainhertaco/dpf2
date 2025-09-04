import pytest
import h5py_stub as h5py

from dpf2.diagnostics.plasma import (
    save_density_temperature_map_hdf5,
    compute_eedf,
    save_eedf_hdf5,
)


def test_save_density_temperature_map_openpmd(tmp_path):
    path = tmp_path / "plasma.h5"
    density = [[1.0]]
    temperature = [[2.0]]
    save_density_temperature_map_hdf5(
        path,
        density,
        temperature,
        response_fn=lambda x: x * 2.0,
        noise_fn=lambda x: 0.1,
        openpmd=True,
    )
    with h5py.File(path, "r") as fh:
        assert fh.attrs["openPMD"] == "1.1.0"
        d_ds = fh["data/0/density"]
        t_ds = fh["data/0/temperature"]
        assert d_ds.data[0][0] == pytest.approx(1.0 * 2.0 + 0.1)
        assert t_ds.data[0][0] == pytest.approx(2.0 * 2.0 + 0.1)


def test_compute_eedf_and_save(tmp_path):
    centers, counts = compute_eedf([0.5, 1.5], [0.0, 1.0, 2.0])
    assert centers == [0.5, 1.5]
    assert counts == [1, 1]
    path = tmp_path / "eedf.h5"
    save_eedf_hdf5(
        path,
        centers,
        counts,
        response_fn=lambda x: x * 2.0,
        noise_fn=lambda x: 0.1,
        openpmd=True,
    )
    with h5py.File(path, "r") as fh:
        assert fh.attrs["openPMD"] == "1.1.0"
        ds = fh["data/0/distribution"]
        assert ds.data[0] == pytest.approx(1 * 2.0 + 0.1)
