import numpy as np
import h5py

from dpf2.eos import TabulatedEOS


def make_table(path):
    rho = np.array([1.0, 2.0])
    e = np.array([10.0, 20.0])
    p = rho[:, None] * e[None, :]
    T = e[None, :] / rho[:, None]
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=rho)
        f.create_dataset("e", data=e)
        f.create_dataset("p", data=p)
        f.create_dataset("T", data=T)


def test_interpolation(tmp_path):
    path = tmp_path / "table.h5"
    make_table(path)
    eos = TabulatedEOS(path)
    rho = np.array([1.5])
    e = np.array([15.0])
    assert np.allclose(eos.pressure(rho, e), rho * e)
    assert np.allclose(eos.temperature(rho, e), e / rho)
