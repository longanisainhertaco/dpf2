import math
import math

from dpf2.diagnostics.neutron_yield import yield_components_with_anisotropy
from dpf2.synthetic_diagnostics.core import tof_iv_lagged_correlation
from dpf2.core.bases import CouplingState


class _AngleEDF:
    def energy_distribution(self, angle_deg: float):
        # Simple distribution varying with angle
        base = 1.0 + angle_deg / 100.0
        return [0.0, 1.0], [base, base]


def test_yield_components_partitioning():
    edf = _AngleEDF()
    cross_section = lambda e: 1.0
    angles = [0.0, 90.0]
    distance = 1.0
    time_bins = [0.0, 1.0]
    reactivity = [1.0, 1.0]
    ion_density = [1.0, 1.0]
    dt = 1.0
    res = yield_components_with_anisotropy(
        edf,
        cross_section,
        angles,
        distance,
        time_bins,
        reactivity,
        ion_density,
        dt,
    )
    assert math.isclose(res["beam_target_total"], sum(res["beam_target"]))
    assert res["thermal_total"] == sum(res["angular_thermal"])
    assert res["angular_thermal"][0] == res["angular_thermal"][1]
    assert "angular_total" in res
    assert res["anisotropy"] > 0.0
    assert "tof_channels" in res and "thermonuclear" in res["tof_channels"]
    assert len(res["tof_phase"]["time_midpoints"]) == len(time_bins) - 1
    assert len(res["tof"]) == len(angles)


def test_tof_iv_phasing():
    history = [CouplingState(current=5.0, voltage=4.0)]
    history += [CouplingState() for _ in range(4)]
    dt = 1e-7
    distance = 0.1
    energies = [0.001]  # 1 keV
    result = tof_iv_lagged_correlation(history, dt, distance, energies)
    lags = result["lags"]
    corr = result["power"]
    idx = corr.index(max(corr))
    best_lag = lags[idx]
    m_n = 1.67492749804e-27
    e_j = energies[0] * 1.602176634e-13
    expected = distance / math.sqrt(2.0 * e_j / m_n)
    assert abs(best_lag - expected) < 2 * dt
