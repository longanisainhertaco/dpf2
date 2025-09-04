from dpf2.diagnostics.neutron_spectra import (
    synthetic_tof_spectrum,
    angular_spectrum,
    anisotropy_metric,
    DetectorLayout,
    load_detector_layout,
    time_resolved_spectra,
    forward_radial_backward_counts,
    anisotropy_ratios,
    cross_correlate_tof_with_circuit,
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


def test_time_resolved_geometry_and_anisotropy(tmp_path):
    import json

    layout_data = {
        "distance_m": 1.0,
        "detectors": [
            {"angle_deg": 0.0, "name": "f"},
            {"angle_deg": 90.0, "name": "r"},
            {"angle_deg": 180.0, "name": "b"},
        ],
    }
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout_data))
    layout = load_detector_layout(layout_path)
    energies = [1.0, 2.0]
    flux = [1.0, 1.0]
    time_bins = [0.0, 1.0]
    spectra = time_resolved_spectra(layout, energies, flux, time_bins)
    assert set(spectra.keys()) == {"f", "r", "b"}
    counts = forward_radial_backward_counts(layout, spectra)
    assert counts["forward"] == counts["radial"] == counts["backward"]
    ratios = anisotropy_ratios(counts)
    assert ratios["forward_backward"] == 1.0


def test_cross_correlation():
    time_bins = [0.0, 1.0, 2.0, 3.0]
    counts = [0.0, 1.0, 0.0]
    circuit_time = [0.5, 1.5, 2.5]
    circuit_signal = [0.0, 1.0, 0.0]
    lags, corr, max_lag = cross_correlate_tof_with_circuit(
        time_bins, counts, circuit_time, circuit_signal
    )
    assert len(lags) == len(corr) == 2 * len(counts) - 1
    assert max_lag == 0.0
