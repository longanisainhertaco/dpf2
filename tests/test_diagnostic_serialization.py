import types
import sys

import numpy as np
import h5py_stub as h5py

# Provide a minimal SciPy stub if the real package is unavailable
try:  # pragma: no cover - optional dependency shim
    import scipy  # type: ignore
except Exception:  # pragma: no cover - exercised when SciPy missing
    _scipy = types.ModuleType("scipy")
    _const = types.SimpleNamespace(c=1, m_n=1, m_e=1, mu_0=1, e=1, epsilon_0=1, k=1)
    _scipy.constants = _const
    _interp = types.ModuleType("scipy.interpolate")
    _interp.interp1d = lambda *a, **k: None
    _scipy.interpolate = _interp
    sys.modules.setdefault("scipy", _scipy)
    sys.modules.setdefault("scipy.constants", _const)
    sys.modules.setdefault("scipy.interpolate", _interp)

# Stub pyevtk which is an optional dependency
sys.modules.setdefault("pyevtk", types.ModuleType("pyevtk"))
sys.modules.setdefault(
    "pyevtk.hl", types.SimpleNamespace(imageToVTK=lambda *a, **k: None)
)

# Augment numpy stub with helpers used by diagnostics
np.ones = lambda shape: np.full(shape, 1.0)
np.trapz = lambda y, dx=1.0: dx * sum(y) if len(y) else 0.0
np.floor = lambda x: int(x)
np.ceil = lambda x: int(x) if x == int(x) else int(x) + 1


def _np_sum(arr):
    try:
        return sum(arr)
    except TypeError:
        return arr


np.sum = _np_sum


class _Linalg:
    @staticmethod
    def norm(v):
        return np.sqrt(np.sum([vi * vi for vi in v]))


np.linalg = _Linalg()


def _histogram(data, bins):
    hist = [0] * (len(bins) - 1)
    for val in data:
        for i in range(len(bins) - 1):
            if bins[i] <= val < bins[i + 1]:
                hist[i] += 1
                break
    return hist, bins


np.histogram = _histogram

from dpf2.simulation.diagnostics import (
    Interferometry,
    XrayDetector,
    NeutronDetector,
)
from dpf2.simulation.utils import FieldManager


def _field_manager():
    return FieldManager((1, 1, 1), 1.0, 1.0, 1.0, (0.0, 0.0, 0.0), {})


def test_interferometry_serialisation(tmp_path):
    fm = _field_manager()
    diag = Interferometry("int", [0, 0, 0], [1, 0, 0], fm)
    state = types.SimpleNamespace(
        density=np.full((1, 1, 1), 1.0), dx=1.0, domain_lo=(0, 0, 0)
    )

    diag.record(0.0, None, None, state=state)
    expected = diag.data[0]["phase_shift"]
    wall = diag.data[0]["wall_time"]

    with h5py.File(tmp_path / "out.h5", "w") as h5:
        diag.to_hdf5(h5)

    with h5py.File(tmp_path / "out.h5", "r") as h5:
        grp = h5["int"]
        assert grp.attrs["openPMD"] == "1.1.0"
        assert np.allclose(grp["time"].data, [0.0])
        assert np.allclose(grp["wall_time"].data, [wall])
        assert np.allclose(grp["phase_shift"].data, [expected])
        assert grp["time"].attrs["unitSI"] == 1.0
        assert grp["phase_shift"].attrs["unitSI"] == 1.0


def test_xray_detector_serialisation(tmp_path):
    fm = _field_manager()
    diag = XrayDetector("xray", [0, 0, 0], fm)
    radiation = types.SimpleNamespace(total_radiated_energy=1.0)
    state = types.SimpleNamespace(
        dx=1.0,
        domain_lo=(0, 0, 0),
        _X=0.5,
        _Y=0.5,
        _Z=0.5,
        cell_volume=1.0,
    )

    diag.record(0.0, None, None, radiation=radiation, state=state)
    expected = diag.data[0]["signal"]
    wall = diag.data[0]["wall_time"]

    with h5py.File(tmp_path / "out.h5", "w") as h5:
        diag.to_hdf5(h5)

    with h5py.File(tmp_path / "out.h5", "r") as h5:
        grp = h5["xray"]
        assert grp.attrs["openPMD"] == "1.1.0"
        assert np.allclose(grp["time"].data, [0.0])
        assert np.allclose(grp["wall_time"].data, [wall])
        assert np.allclose(grp["signal"].data, [expected])
        assert grp["energy_bins"].data == diag.energy_bins
        assert grp["signal"].attrs["unitSI"] == 1.0


def test_neutron_detector_serialisation(tmp_path):
    fm = _field_manager()
    bins = [0.0, 1.0, 2.0]
    diag = NeutronDetector("nd", [0, 0, 0], bins, fm, reaction="DD")
    pic = types.SimpleNamespace(
        get_neutron_events=lambda reaction: [
            {"position": [0, 0, 0], "energy": 10.0, "time": 0.0}
        ]
    )

    diag.record(0.0, None, None, pic=pic)
    expected = diag.data[0]["histogram"]
    wall = diag.data[0]["wall_time"]

    with h5py.File(tmp_path / "out.h5", "w") as h5:
        diag.to_hdf5(h5)

    with h5py.File(tmp_path / "out.h5", "r") as h5:
        grp = h5["nd"]
        assert grp.attrs["openPMD"] == "1.1.0"
        assert np.allclose(grp["time"].data, [0.0])
        assert np.allclose(grp["wall_time"].data, [wall])
        assert np.array(grp["histogram"].data).shape == (1, len(bins) - 1)
        assert np.allclose(grp["histogram"].data[0], expected)
        assert np.allclose(grp["time_bins"].data, bins)
        assert grp["histogram"].attrs["unitSI"] == 1.0
        assert grp.attrs["reaction"] == "DD"
