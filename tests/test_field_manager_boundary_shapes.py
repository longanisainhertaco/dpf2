import numpy as np
from dpf2.simulation.utils import FieldManager

def test_apply_boundary_conditions_preserves_shape_and_updates_boundaries():
    nx = ny = nz = 4
    boundary_conditions = {
        "x_lo": "dirichlet",
        "x_hi": "dirichlet",
        "y_lo": "dirichlet",
        "y_hi": "dirichlet",
        "z_lo": "dirichlet",
        "z_hi": "dirichlet",
        "ghost_cells": 1,
    }
    fm = FieldManager(
        grid_shape=(nx, ny, nz),
        dx=1.0,
        dy=1.0,
        dz=1.0,
        domain_lo=(0.0, 0.0, 0.0),
        boundary_conditions=boundary_conditions,
    )
    fm.E.fill(1.0)
    fm.B.fill(1.0)
    fm.J.fill(1.0)

    fm.apply_boundary_conditions(None)

    expected_shape = (3, nx, ny, nz)
    assert fm.E.shape == expected_shape
    assert fm.B.shape == expected_shape
    assert fm.J.shape == expected_shape

    for arr in (fm.E, fm.B, fm.J):
        # Boundary cells should be zero due to dirichlet conditions
        assert np.all(arr[:, 0, :, :] == 0.0)
        assert np.all(arr[:, -1, :, :] == 0.0)
        assert np.all(arr[:, :, 0, :] == 0.0)
        assert np.all(arr[:, :, -1, :] == 0.0)
        assert np.all(arr[:, :, :, 0] == 0.0)
        assert np.all(arr[:, :, :, -1] == 0.0)
        # Interior cells remain unchanged
        assert np.all(arr[:, 1:-1, 1:-1, 1:-1] == 1.0)
