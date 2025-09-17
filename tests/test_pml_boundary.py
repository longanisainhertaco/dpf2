import numpy as np
from dpf2.simulation.utils import FieldManager


def test_pml_absorbs_outgoing_wave():
    nx = 10
    boundary_conditions = {
        "x_lo": "outflow",
        "x_hi": "pml",
        "y_lo": "periodic",
        "y_hi": "periodic",
        "z_lo": "periodic",
        "z_hi": "periodic",
        "pml_thickness": 2,
        "pml_sigma": 1.0,
        "pml_profile": "exponential",
    }

    fm = FieldManager(
        grid_shape=(nx, 1, 1),
        dx=1.0,
        dy=1.0,
        dz=1.0,
        domain_lo=(0.0, 0.0, 0.0),
        boundary_conditions=boundary_conditions,
    )

    x = np.arange(nx)
    fm.E[0, :, 0, 0] = np.sin(2 * np.pi * x / nx)
    fm.B[1, :, 0, 0] = np.sin(2 * np.pi * x / nx)

    interior_E_before = fm.E[0, -3, 0, 0]
    interior_B_before = fm.B[1, -3, 0, 0]
    fm.apply_boundary_conditions(None)

    # Expect exponential damping in the ghost cells
    expected_1 = interior_E_before * np.exp(-boundary_conditions["pml_sigma"] * 1)
    expected_2 = interior_E_before * np.exp(-boundary_conditions["pml_sigma"] * 2)
    expected_B1 = interior_B_before * np.exp(-boundary_conditions["pml_sigma"] * 1)
    expected_B2 = interior_B_before * np.exp(-boundary_conditions["pml_sigma"] * 2)

    assert np.isclose(fm.E[0, -2, 0, 0], expected_1)
    assert np.isclose(fm.E[0, -1, 0, 0], expected_2)
    assert np.isclose(fm.B[1, -2, 0, 0], expected_B1)
    assert np.isclose(fm.B[1, -1, 0, 0], expected_B2)

    # Ensure interior cell is unchanged (no reflection)
    assert np.isclose(fm.E[0, -3, 0, 0], interior_E_before)
    assert np.isclose(fm.B[1, -3, 0, 0], interior_B_before)
