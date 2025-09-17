import math
import math
from pathlib import Path

import h5py_stub as h5py  # type: ignore

from dpf2.diagnostics import (
    apply_detector_response,
    apply_noise,
    interferometer_phase_shift,
    pinhole_image,
    save_anisotropic_spectrum_hdf5,
)


def test_interferometer_phase_shift():
    ne = [1e19, 2e19]
    dl = [0.01, 0.01]
    wavelength = 1e-6
    r_e = 2.8179403262e-15
    expected = -r_e * wavelength * (1e19 * 0.01 + 2e19 * 0.01)
    result = interferometer_phase_shift(ne, dl, wavelength)
    assert math.isclose(result, expected)


def test_pinhole_image_single_source():
    positions = [(0.0, 0.0, 1.0)]
    intensities = [1.0]
    img = pinhole_image(positions, intensities, 1.0, (10, 10), 0.1)
    assert img[5][5] > 0.0
    assert all(
        img[j][i] == 0.0 for j in range(10) for i in range(10) if (i, j) != (5, 5)
    )


def test_noise_and_response_functions():
    signal = [0.0, 0.0]
    noisy = apply_noise(signal, noise_fn=lambda: 1.0)
    assert noisy == [1.0, 1.0]
    responded = apply_detector_response(signal, response_fn=lambda x: x * 2.0)
    assert responded == [0.0, 0.0]
    responded = apply_detector_response([1.0], response_fn=lambda x: x * 2.0)
    assert responded == [2.0]


def test_save_anisotropic_spectrum_hdf5(tmp_path: Path):
    energies = [1.0, 2.0]
    angles = [0.0, 90.0]
    spectrum = [[1.0, 0.5], [0.2, 0.1]]
    fname = tmp_path / "spec.h5"
    save_anisotropic_spectrum_hdf5(fname, energies, angles, spectrum, ["d1", "d2"])
    with h5py.File(fname, "r") as fh:
        assert "detectors" in fh._items
        grp = fh["detectors"]
        assert "d1" in grp._items and "d2" in grp._items
