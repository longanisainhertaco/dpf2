import numpy as np
from Simulation.utils import FieldManager

def test_deposit_charge_updates_rho():
    grid_shape = (4, 4, 4)
    fm = FieldManager(grid_shape, 1.0, 1.0, 1.0, (0, 0, 0), {})
    initial_rho = np.copy(fm.rho)
    assert np.all(initial_rho == 0)
    charge = np.ones(grid_shape)
    fm.deposit_charge(charge)
    assert np.allclose(fm.rho, charge)
    fm.deposit_charge(charge)
    assert np.allclose(fm.rho, 2 * charge)
