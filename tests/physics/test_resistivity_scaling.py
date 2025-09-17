import numpy as np

from dpf2.hall_mhd_solver import spitzer_resistivity


def test_spitzer_resistivity_temperature_scaling():
    """Doubling ``T_e`` should reduce ``η`` by ``2^{-3/2}``."""
    ne = 1e20  # m^-3
    Te = 1e5  # K
    Z = 1.0

    eta1 = spitzer_resistivity(ne, Te, Z)
    eta2 = spitzer_resistivity(ne, 2 * Te, Z)

    expected = eta1 / (2**1.5)
    assert np.isclose(eta2, expected)
