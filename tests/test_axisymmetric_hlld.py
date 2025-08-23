import numpy as np

from dpf2.solvers.axisymmetric_hlld import AxisymmetricHLLD


def divergence(B_r: np.ndarray, B_z: np.ndarray, dr: float, dz: float) -> np.ndarray:
    """Compute discrete divergence of the magnetic field."""
    return (
        (B_r[2:, 1:-1] - B_r[:-2, 1:-1]) / (2.0 * dr)
        + (B_z[1:-1, 2:] - B_z[1:-1, :-2]) / (2.0 * dz)
    )


def test_constrained_transport_divergence_free():
    nr = nz = 32
    dr = dz = 1.0 / nr
    r = np.linspace(0.0, 1.0, nr)
    z = np.linspace(0.0, 1.0, nz)
    R, Z = np.meshgrid(r, z, indexing="ij")

    # Divergence free field defined via a stream function
    psi = np.sin(np.pi * R) * np.sin(np.pi * Z)
    B_r = -np.gradient(psi, dz, axis=1)
    B_z = np.gradient(psi, dr, axis=0)

    # Initial density, velocity and energy
    rho = np.ones_like(B_r)
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
