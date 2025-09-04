import numpy as np
import h5py_stub as h5py

from dpf2.diagnostics.neutron_yield import (
    IonBeamEDF,
    compute_beam_target_yield,
    compute_thermonuclear_yield,
    save_anisotropic_spectrum_hdf5,
    yield_components_with_anisotropy,
    tof_iv_cross_correlation,
)
from dpf2.diagnostics.detector_models import (
    cr39_response,
    rcf_response,
    time_gated_scintillator_response,
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


def test_yield_components_and_anisotropy():
    beam = _TestBeam()
    cross_section = lambda e: e
    angles = [0.0, 90.0]
    distance = 1.0
    time_bins = [0.0, 1.0]
    reactivity = [1e-20, 1e-20]
    ion_density = [1e19, 1e19]
    dt = 1e-6
    result = yield_components_with_anisotropy(
        beam,
        cross_section,
        angles,
        distance,
        time_bins,
        reactivity,
        ion_density,
        dt,
    )
    bt, _ = compute_beam_target_yield(beam, cross_section, angles, distance, time_bins)
    th = compute_thermonuclear_yield(reactivity, ion_density, dt)
    th_per = [th / 2.0, th / 2.0]
    total = [b + t for b, t in zip(bt, th_per)]
    mean = sum(total) / 2.0
    expected_aniso = (max(total) - min(total)) / mean if mean else 0.0
    assert result["beam_target"] == bt
    assert result["thermonuclear"] == th
    assert result["angular_thermal"] == th_per
    assert result["anisotropy"] == expected_aniso


def test_detector_models():
    yields = [10.0, 20.0]
    area = 1e-4
    distance = 1.0
    expected = [y * area / distance ** 2 for y in yields]
    assert cr39_response(yields, area, distance) == expected
    assert rcf_response(yields, area, distance) == expected

    hist = [1.0, 2.0, 3.0]
    bins = [0.0, 1.0, 2.0, 3.0]
    count = time_gated_scintillator_response(hist, bins, 0.0, 3.0, area, distance)
    assert count == sum(hist) * area / distance ** 2


def test_tof_iv_cross_correlation():
    tof = [1.0, 2.0, 3.0]
    current = [1.0, 2.0, 3.0]
    voltage = [3.0, 2.0, 1.0]
    corr = tof_iv_cross_correlation(tof, current, voltage)
    assert np.isclose(corr["current"], 1.0)
    assert np.isclose(corr["voltage"], -1.0)
