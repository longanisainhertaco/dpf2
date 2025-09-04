import math

from dpf2.neutron_yield_model import TabulatedIonEDF
from dpf2.diagnostics import (
    simulate_tof_detectors,
    save_tof_hdf5,
    angular_yield_map,
    save_angular_yield_map_hdf5,
)


def _make_edf():
    E1 = 1e-13
    E2 = 1.5e-13
    data = {
        0.0: ([E1, E2], [2.0, 0.0]),
        90.0: ([E1, E2], [1.0, 0.0]),
    }
    return TabulatedIonEDF(data), E1, E2


def test_forward_vs_radial_counts(tmp_path):
    edf, E1, E2 = _make_edf()
    cross_section = lambda e: 1.0
    distance = 1.0
    m_n = 1.674e-27
    E_mid = (E1 + E2) / 2.0
    t_exp = distance / math.sqrt(2.0 * E_mid / m_n)
    time_bins = [0.0, t_exp * 0.9, t_exp * 1.1]
    angles = [0.0, 90.0]

    dets = simulate_tof_detectors(edf, cross_section, angles, distance, time_bins)
    assert sum(dets["detector_0"]) > sum(dets["detector_1"])
    tof_path = tmp_path / "tof.h5"
    save_tof_hdf5(tof_path, time_bins, dets)
    assert tof_path.exists()

    energy_bins = [E1, E2]
    spec = angular_yield_map(edf, cross_section, angles, energy_bins)
    assert sum(spec[0]) > sum(spec[1])
    spec_path = tmp_path / "spec.h5"
    save_angular_yield_map_hdf5(spec_path, energy_bins, angles, spec)
    assert spec_path.exists()
