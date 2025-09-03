import numpy as np
import h5py_stub as h5py

from dpf2.diagnostics.neutron_yield import (
    compute_beam_target_yield,
    compute_thermonuclear_yield,
    save_anisotropic_spectrum_hdf5,
)


def test_compute_beam_and_thermonuclear_yields():
    ion_dist = [1.0, 2.0]
    target_density = [1e19, 1e19]
    cross_section = [1e-24, 2e-24]
    dt = 1e-6
    bt = compute_beam_target_yield(ion_dist, target_density, cross_section, dt)
    expected_bt = sum(i * n * s for i, n, s in zip(ion_dist, target_density, cross_section)) * dt
    assert bt == expected_bt

    reactivity = [1e-20, 2e-20]
    ion_density = [1e19, 1e19]
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
        np.testing.assert_allclose(fh["spectrum"][:], spectrum)
