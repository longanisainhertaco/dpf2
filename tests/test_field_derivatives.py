import numpy as np
from Simulation.utils import FieldManager


def test_divergence_and_curl_periodic():
    nx = ny = nz = 16
    L = 2 * np.pi
    dx = dy = dz = L / nx
    boundaries = {f"{ax}_{side}": "periodic" for ax in ['x', 'y', 'z'] for side in ['lo', 'hi']}
    fm = FieldManager((nx, ny, nz), dx, dy, dz, (0.0, 0.0, 0.0), boundaries)
    x = np.linspace(0, L, nx, endpoint=False)
    y = np.linspace(0, L, ny, endpoint=False)
    z = np.linspace(0, L, nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    field_div = np.zeros((3, nx, ny, nz))
    field_div[0] = np.sin(X)
    field_div[1] = np.sin(Y)
    field_div[2] = np.sin(Z)
    div_numeric = fm.compute_divergence(field_div)
    div_analytic = np.cos(X) + np.cos(Y) + np.cos(Z)
    assert np.allclose(div_numeric, div_analytic, atol=1e-1)

    field_curl = np.zeros((3, nx, ny, nz))
    field_curl[0] = np.sin(Y)
    field_curl[1] = np.sin(Z)
    field_curl[2] = np.sin(X)
    curl_numeric = fm.compute_curl(field_curl)
    curl_analytic = np.array([
        -np.cos(Z),
        -np.cos(X),
        -np.cos(Y),
    ])
    assert np.allclose(curl_numeric, curl_analytic, atol=1e-1)


def test_dirichlet_boundary_handling():
    nx, ny, nz = 16, 4, 4
    L = 1.0
    dx = L / (nx - 1)
    dy = dz = 1.0
    boundaries = {
        'x_lo': 'dirichlet', 'x_hi': 'dirichlet',
        'y_lo': 'periodic', 'y_hi': 'periodic',
        'z_lo': 'periodic', 'z_hi': 'periodic'
    }
    fm = FieldManager((nx, ny, nz), dx, dy, dz, (0.0, 0.0, 0.0), boundaries)
    x = np.linspace(0, L, nx)
    X = x[:, None, None]

    field = np.zeros((3, nx, ny, nz))
    field[0] = np.sin(np.pi * X / L)
    div_numeric = fm.compute_divergence(field)
    div_analytic = (np.pi / L) * np.cos(np.pi * X / L)
    assert np.allclose(div_numeric, div_analytic, atol=1e-1)
