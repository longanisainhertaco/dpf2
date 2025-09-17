import numpy as np
import pytest
from dpf2.solvers.axisymmetric_hlld import AxisymmetricHLLD


def divergence(B_r, B_z, dr: float, dz: float):
    """Compute discrete divergence of the magnetic field."""
    return (B_r[2:, 1:-1] - B_r[:-2, 1:-1]) / (2.0 * dr) + (
        B_z[1:-1, 2:] - B_z[1:-1, :-2]
    ) / (2.0 * dz)


@pytest.mark.skipif(not hasattr(np, "meshgrid"), reason="requires full numpy")
def test_constrained_transport_divergence_free():
    nr = nz = 32
    dr = dz = 1.0 / nr
    r = np.linspace(0.0, 1.0, nr)
    z = np.linspace(0.0, 1.0, nz)
    R = np.array([[ri for _ in z] for ri in r])
    Z = np.array([[zj for zj in z] for _ in r])

    # Start with a trivial divergence free magnetic field
    B_r = np.zeros((nr, nz))
    B_z = np.zeros((nr, nz))

    # Initial density, velocity and energy
    rho = np.full(B_r.shape, 1.0)
    v_r = np.sin(2.0 * np.pi * Z)
    mom_r = rho * v_r
    mom_phi = np.zeros_like(B_r)
    mom_z = np.zeros_like(B_r)
    B_phi = np.zeros_like(B_r)

    gamma = 5.0 / 3.0
    p0 = 1.0
    v_sq = v_r**2
    energy = p0 / (gamma - 1.0) + 0.5 * rho * v_sq + 0.5 * (B_r**2 + B_z**2)

    state = {
        "rho": rho,
        "mom_r": mom_r,
        "mom_phi": mom_phi,
        "mom_z": mom_z,
        "B_r": B_r.copy(),
        "B_phi": B_phi,
        "B_z": B_z.copy(),
        "energy": energy,
    }

    solver = AxisymmetricHLLD(gamma=gamma)

    div_before = divergence(state["B_r"], state["B_z"], dr, dz)
    assert np.allclose(div_before, 0.0, atol=1e-12)

    solver.step(state, dt=1e-3, dr=dr, dz=dz)

    div_after = divergence(state["B_r"], state["B_z"], dr, dz)
    # Constrained transport keeps the discrete divergence small.
    assert np.max(np.abs(div_after)) < 2e-2


def test_source_terms_added_to_state():
    grid = (3, 3)
    zeros = np.zeros(grid)
    state = {
        "rho": np.full(grid, 1.0),
        "mom_r": zeros.copy(),
        "mom_phi": zeros.copy(),
        "mom_z": zeros.copy(),
        "B_r": zeros.copy(),
        "B_phi": zeros.copy(),
        "B_z": zeros.copy(),
        "energy": np.full(grid, 1.0),
    }
    solver = AxisymmetricHLLD()
    src = {
        "rho": np.full(grid, 1.0),
        "energy": np.full(grid, 2.0),
        "A": np.full(grid, 1.0),
    }
    solver.step(state, dt=0.5, sources=src, sources_only=True)
    assert np.allclose(state["rho"], np.full(grid, 1.5))
    assert np.allclose(state["energy"], np.full(grid, 2.0))
    assert "A" in state and np.allclose(state["A"], np.full(grid, 0.5))
