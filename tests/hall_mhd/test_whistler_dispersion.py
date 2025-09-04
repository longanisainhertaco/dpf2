import numpy as np

from dpf2.physics.hall_mhd import whistler_dispersion, mu_0, q_e, m_p


def test_whistler_dispersion_matches_analytic():
    k = 5.0  # 1/m
    ne = 1e19  # m^-3
    B = 2.0  # Tesla
    omega = whistler_dispersion(k, ne, B)

    di = np.sqrt(m_p / (mu_0 * ne * q_e ** 2))
    omega_ci = abs(q_e) * B / m_p
    analytic = omega_ci * (k * di) ** 2
    assert np.isclose(omega, analytic)
