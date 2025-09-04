import math
import pytest
import h5py_stub as h5py

from dpf2.diagnostics.pinhole_imaging import pinhole_image
from dpf2.diagnostics.synthetic_signals import current_waveform
from dpf2.diagnostics.neutron_yield import save_anisotropic_spectrum_hdf5
from dpf2.core.bases import CouplingState


def test_noise_response_pinhole():
    img = pinhole_image(
        [(0.0, 0.0, 1.0)],
        [1.0],
        detector_distance=1.0,
        detector_pixels=(1, 1),
        pixel_size=1.0,
        response_fn=lambda x: x * 2.0,
        noise_fn=lambda x: 0.1,
    )
    expected = (1.0 / (4.0 * math.pi)) * 2.0 + 0.1
    assert abs(img[0][0] - expected) < 1e-12


def test_noise_response_synthetic_signal():
    history = [CouplingState(current=1.0), CouplingState(current=2.0)]
    data = current_waveform(
        history,
        response_fn=lambda x: x * 2.0,
        noise_fn=lambda x: 0.1,
    )
    assert data == [1.0 * 2.0 + 0.1, 2.0 * 2.0 + 0.1]


def test_save_anisotropic_spectrum_openpmd(tmp_path):
    path = tmp_path / "spec.h5"
    energies = [1.0, 2.0]
    angles = [0.0, 45.0]
    spectrum = [[1.0, 2.0], [3.0, 4.0]]
    save_anisotropic_spectrum_hdf5(
        path,
        energies,
        angles,
        spectrum,
        response_fn=lambda x: x * 2.0,
        noise_fn=lambda x: 0.1,
        openpmd=True,
    )
    with h5py.File(path, "r") as fh:
        assert fh.attrs["openPMD"] == "1.1.0"
        ds = fh["data/0/detectors/detector_0"]
        data = list(ds.data)
        assert data[0] == pytest.approx(1.0 * 2.0 + 0.1)
        assert ds.attrs["unitSI"] == 1.0
