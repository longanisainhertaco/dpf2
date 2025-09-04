import numpy as np
import h5py_stub as h5py

from dpf2.diagnostics.neutron_yield import (
    IonBeamEDF,
    compute_beam_target_yield,
    compute_thermonuclear_yield,
    save_anisotropic_spectrum_hdf5,
)


class _TestBeam(IonBeamEDF):
    def energy_distribution(self, angle_deg: float):
        return [1.0, 2.0], [1.0, 1.0]


def test_compute_beam_and_thermonuclear_yields():
    beam = _TestBeam()
    cross_section = lambda e: e
    angles = [0.0]
    distance = 1.0
    time_bins = [0.0, 1.0]
    yield_vals, _ = compute_beam_target_yield(beam, cross_section, angles, distance, time_bins)
    expected_bt = 0.5 * ((1.0 * 1.0) + (1.0 * 2.0)) * (2.0 - 1.0)
    assert yield_vals[0] == expected_bt

    reactivity = [1e-20, 2e-20]
    ion_density = [1e19, 1e19]
    dt = 1e-6
    th = compute_thermonuclear_yield(reactivity, ion_density, dt)
    expected_th = sum(r * n ** 2 for r, n in zip(reactivity, ion_density)) * dt
    assert th == expected_th


def test_save_anisotropic_spectrum_hdf5(tmp_path):
    energies = [1.0, 2.0]
    angles = [0.0, 45.0, 90.0]
    spectrum = [[1, 2], [3, 4], [5, 6]]
    path = tmp_path / "spec.h5"
    save_anisotropic_spectrum_hdf5(path, energies, angles, spectrum)
    with h5py.File(path, "r") as fh:
        np.testing.assert_allclose(fh["energy_MeV"][:], energies)
        np.testing.assert_allclose(fh["angle_deg"][:], angles)
        for i, row in enumerate(spectrum):
            np.testing.assert_allclose(fh[f"detectors/detector_{i}"][:], row)
