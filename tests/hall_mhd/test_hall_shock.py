import numpy as np

from dpf2.physics.hall_mhd import hall_shock_speed, mu_0, m_p, q_e


def test_hall_shock_speed_scaling():
    ne = 1e19
    B = 1.0
    L = 0.1
    speed = hall_shock_speed(B, ne, L)

    vA = B / np.sqrt(mu_0 * m_p * ne)
    di = np.sqrt(m_p / (mu_0 * ne * q_e**2))
    expected = vA * (1.0 + di / L)
    assert np.isclose(speed, expected)
