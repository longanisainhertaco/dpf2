import numpy as np
import importlib.util
from pathlib import Path

# Import FieldManager directly from the source file to avoid triggering package
# level imports that require heavy dependencies.
utils_path = Path(__file__).resolve().parents[1] / "src" / "dpf2" / "simulation" / "utils.py"
spec = importlib.util.spec_from_file_location("field_utils", utils_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FieldManager = module.FieldManager

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


def test_apply_boundary_conditions_periodic_wraps_values():
    nx = ny = nz = 4
    boundary_conditions = {
        "x_lo": "periodic",
        "x_hi": "periodic",
        "y_lo": "periodic",
        "y_hi": "periodic",
        "z_lo": "periodic",
        "z_hi": "periodic",
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

    # Populate fields with distinct values along the x-axis
    for arr in (fm.E, fm.B, fm.J):
        for i in range(nx):
            arr[:, i, :, :] = i

    fm.apply_boundary_conditions(None)

    expected_shape = (3, nx, ny, nz)
    assert fm.E.shape == expected_shape
    assert fm.B.shape == expected_shape
    assert fm.J.shape == expected_shape

    for arr in (fm.E, fm.B, fm.J):
        # Ghost cells should wrap to the opposite interior values
        assert np.all(arr[:, 0, :, :] == arr[:, -2, :, :])
        assert np.all(arr[:, -1, :, :] == arr[:, 1, :, :])
