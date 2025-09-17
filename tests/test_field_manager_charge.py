import numpy as np
from dpf2.simulation.utils import FieldManager


def test_deposit_charge_updates_rho():
    grid_shape = (4, 4, 4)
    fm = FieldManager(grid_shape, 1.0, 1.0, 1.0, (0, 0, 0), {})
    assert np.all(fm.rho == 0)
    charge1 = np.ones(grid_shape)
    charge2 = 2 * np.ones(grid_shape)
    fm.deposit_charge(charge1)
    fm.deposit_charge(charge2)
    total_charge = charge1 + charge2
    assert np.allclose(fm.rho, total_charge)

    # Check that rho is included in checkpoint data
    checkpoint = fm.checkpoint()
    assert "rho" in checkpoint
    assert np.allclose(np.array(checkpoint["rho"]), total_charge)

    # Check that diagnostics report rho
    diagnostics = fm.get_diagnostics()
    assert "rho" in diagnostics
    assert np.allclose(np.array(diagnostics["rho"]), total_charge)
