from dpf2.diagnostics.neutron_spectra import (
    synthetic_tof_spectrum,
    angular_spectrum,
    anisotropy_metric,
    DetectorLayout,
)


def test_synthetic_tof_spectrum():
    energies = [1.0, 2.0]
    flux = [1.0, 1.0]
    distance = 1.0
    time_bins = [0.0, 1.0]
    hist = synthetic_tof_spectrum(energies, flux, distance, time_bins)
    assert len(hist) == 1
    assert hist[0] == 1.0


def test_angular_spectrum_and_anisotropy():
    angles = [0.0, 90.0, 180.0]
    spectrum = angular_spectrum(angles, base_yield=1.0, anisotropy=1.0)
    assert spectrum == [2.0, 1.0, 0.0]
    metric = anisotropy_metric(spectrum)
    assert metric == 2.0
    layout = DetectorLayout(angles=angles, distance_m=1.0)
    assert layout.detectors[1].angle_deg == 90.0
