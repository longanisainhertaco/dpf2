import math

from dpf2.diagnostics.neutron_yield import IonBeamEDF, compute_beam_target_yield


class _Beam(IonBeamEDF):
    def __init__(self, data):
        self.data = data

    def energy_distribution(self, angle_deg: float):
        return self.data[angle_deg]


def test_forward_anisotropy_and_tof():
    E1 = 1e-13
    E2 = 1.5e-13
    beam = _Beam(
        {
            0.0: ([E1, E2], [2.0, 0.0]),
            90.0: ([E1, E2], [1.0, 0.0]),
        }
    )
    cross_section = lambda e: 1.0
    distance = 1.0
    E_mid = (E1 + E2) / 2.0
    m_n = 1.674e-27
    t_exp = distance / math.sqrt(2.0 * E_mid / m_n)
    time_bins = [0.0, t_exp * 0.9, t_exp * 1.1]
    angles = [0.0, 90.0]
    yields, tofs = compute_beam_target_yield(
        beam, cross_section, angles, distance, time_bins
    )
    assert yields[0] > yields[1]
    assert tofs[0][0] == 0.0
    assert tofs[0][1] > 0.0
    assert tofs[1][1] == tofs[0][1] / 2.0
