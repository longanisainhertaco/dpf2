from dpf2.diagnostics.neutron import (
    thermonuclear_yield,
    beam_target_yield,
    angular_distribution,
    synthetic_tof_correlated,
    compare_with_benchmark,
)


class _ConstEDF:
    def energy_distribution(self, angle_deg):
        return [1.0, 2.0], [1.0, 1.0]


def _unit_sigma(e):
    return 1.0


def test_thermonuclear_yield():
    y = thermonuclear_yield([1.0, 1.0], [1.0, 2.0], 1.0)
    assert y == 5.0


def test_beam_target_yield_basic():
    angles = [0.0]
    time_bins = [0.0, 1.0e-12]
    yields, tofs = beam_target_yield(_ConstEDF(), _unit_sigma, angles, 1.0, time_bins)
    assert yields == [1.0]
    assert len(tofs) == 1 and len(tofs[0]) == 1


def test_angular_distribution_counts():
    spectrum, counts = angular_distribution([0.0, 90.0, 180.0], 1.0, anisotropy=1.0)
    assert spectrum == [2.0, 1.0, 0.0]
    assert counts["forward"] == 2.0
    assert counts["radial"] == 1.0
    assert counts["backward"] == 0.0


def test_synthetic_tof_correlated():
    energies = [3e-22, 4e-22]
    flux = [1.0e22, 1.0e22]
    distance = 1.0
    time_bins = [0.0, 0.001, 0.002, 0.003]
    circuit_time = [0.0, 0.001, 0.002, 0.003]
    current = [0.0, 1.0, 0.0, 0.0]
    voltage = [0.0, 2.0, 0.0, 0.0]
    hist, peaks, max_lag = synthetic_tof_correlated(
        energies, flux, distance, time_bins, circuit_time, current, voltage
    )
    expected = [0.0, 1.0, 0.0]
    for a, b in zip(hist, expected):
        assert abs(a - b) < 1e-12
    assert peaks == [(0.0015, 1.0)]
    assert max_lag == 0.0


def test_compare_with_benchmarks():
    passed, diff = compare_with_benchmark(0.0, "pf_1000")
    assert passed and diff == 0.0
    passed, _ = compare_with_benchmark(1.0, "mjolnir", pass_band=2.0)
    assert passed
