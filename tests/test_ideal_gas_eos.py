import numpy as np

from dpf2.eos.ideal_gas import IdealGasEOS


def test_ideal_gas_basic_relations():
    eos = IdealGasEOS(gamma=1.4, mu=2.0, ionization=1.0)
    rho = np.array([1.0])
    T = np.array([3.0])
    R = 1.0 / 2.0
    p_i = eos.ion_pressure(rho, T)
    p_e = eos.electron_pressure(rho, T)
    e_i = eos.ion_energy(rho, T)
    e_e = eos.electron_energy(rho, T)
    assert np.allclose(p_i, rho * R * T)
    assert np.allclose(p_e, rho * R * T)
    expected_e = R * T / (1.4 - 1.0)
    assert np.allclose(e_i, expected_e)
    assert np.allclose(e_e, expected_e)
