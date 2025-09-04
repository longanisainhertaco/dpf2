import numpy as np

from dpf2.eos.ideal_gas import IdealGasEOS


def test_ideal_gas_basic_relations():
    eos = IdealGasEOS(gamma=1.4, mu=2.016, ionization=1.0)
    rho = np.array([0.0899])  # kg/m^3 for H2 at STP
    T = np.array([273.15])  # Kelvin
    R_specific = 8.31446261815324 * 1000.0 / 2.016
    p_i = eos.ion_pressure(rho, T)
    p_e = eos.electron_pressure(rho, T)
    e_i = eos.ion_energy(rho, T)
    e_e = eos.electron_energy(rho, T)
    expected_p = rho * R_specific * T
    expected_e = R_specific * T / (1.4 - 1.0)
    assert np.allclose(p_i, expected_p)
    assert np.allclose(p_e, expected_p)
    assert np.allclose(e_i, expected_e)
    assert np.allclose(e_e, expected_e)
    assert np.isclose(p_i[0], 101275.53681892625, rtol=1e-5)
    assert np.isclose(e_i[0], 2816338.62121597, rtol=1e-5)
