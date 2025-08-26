import numpy as np
import h5py_stub as h5py

from dpf2.eos import TabulatedEOS


def make_table(path):
    rho = np.array([1.0, 2.0])
    T = np.array([10.0, 20.0])
    p = rho[:, None] * T[None, :]
    e = T[None, :] / rho[:, None]
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=rho)
        f.create_dataset("T", data=T)
        f.create_dataset("p", data=p)
        f.create_dataset("e", data=e)


def test_interpolation(tmp_path):
    path = tmp_path / "table.h5"
    make_table(path)
    eos = TabulatedEOS(path)
    rho = np.array([1.5])
    T = np.array([15.0])
    assert np.allclose(eos.pressure(rho, T), rho * T)
    # Linear interpolation of ``e = T / rho`` on the tabulated grid yields
    # ``11.25`` at ``rho=1.5`` and ``T=15``.  This verifies that the fallback
    # interpolator produces sensible results without requiring SciPy.
    assert np.allclose(eos.energy(rho, T), np.array([11.25]))
