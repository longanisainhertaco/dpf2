import numpy as np
import pytest
from neutron_yield_model import (
    BoschHaleTable,
    DEFAULT_BOSCH_HALE_TABLE,
    compute_dd_yield,
)


def test_bosch_hale_yield_matches_manual_calculation():
    table = BoschHaleTable.from_csv(DEFAULT_BOSCH_HALE_TABLE)
    t = np.linspace(0.0, 1e-6, 5)
    T = np.full_like(t, 100.0)  # keV
    n = np.full_like(t, 1e20)  # m^-3
    volume = np.full_like(t, 1e-12)
    result = compute_dd_yield(t, T, n, volume, table)

    # manual calculation for benchmark
    energy_MeV = T * 1e-3
    sigma_m2 = table.sigma(energy_MeV) * 1e-28
    E_J = energy_MeV * 1e6 * 1.602e-19
    v_rel = np.sqrt(2.0 * E_J / 3.344e-27)
    reactivity = sigma_m2 * v_rel
    rate = 0.25 * n**2 * reactivity * volume
    expected = np.trapz(rate, t)
    assert result == pytest.approx(expected)
