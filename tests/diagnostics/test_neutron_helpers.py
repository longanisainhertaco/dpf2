import math

from dpf2.diagnostics.neutron import (
    synthetic_tof_spectrum,
    angular_spectrum,
    anisotropy_report,
    DetectorLayout,
)


def test_basic_wrappers_and_anisotropy():
    energies = [1.0, 2.0]
    flux = [1.0, 1.0]
    distance = 1.0
    time_bins = [0.0, 1.0]
    hist = synthetic_tof_spectrum(energies, flux, distance, time_bins)
    assert hist == [1.0]

    spectrum = angular_spectrum([0.0, 90.0, 180.0], base_yield=1.0, anisotropy=1.0)
    assert spectrum == [2.0, 1.0, 0.0]

    layout = DetectorLayout(
        angles=[0.0, 90.0, 180.0], distance_m=1.0, names=["f", "r", "b"]
    )
    spectra = {"f": [1.0], "r": [1.0], "b": [1.0]}
    report = anisotropy_report(layout, spectra)
    counts = report["counts"]
    assert counts["forward"] == counts["radial"] == counts["backward"]
    assert math.isclose(report["metric"], 0.0)
    assert report["ratios"]["forward_backward"] == 1.0
