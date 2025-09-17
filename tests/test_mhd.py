import numpy as np
import sys
from pathlib import Path

from dpf2.physics import ResistiveMHD


def test_conservative_variables():
    model = ResistiveMHD()
    primitives = np.array([1.0, 2.0, -1.0, 0.5, 0.5, 0.1, 0.2, 0.3])
    U = model.conservative_variables(primitives)

    rho, vx, vy, vz, p, Bx, By, Bz = primitives
    kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    magnetic = 0.5 * (Bx**2 + By**2 + Bz**2)
    energy = p / (model.gamma - 1.0) + kinetic + magnetic
    expected = np.array([rho, rho * vx, rho * vy, rho * vz, energy, Bx, By, Bz, 0.0])

    assert np.allclose(U, expected)


def test_flux_function():
    model = ResistiveMHD()
    primitives = np.array([1.0, 2.0, -1.0, 0.5, 0.5, 0.1, 0.2, 0.3])
    U = model.conservative_variables(primitives)

    rho, vx, vy, vz, p, Bx, By, Bz = primitives
    B2 = Bx**2 + By**2 + Bz**2
    total_p = p + 0.5 * B2
    Bdotv = Bx * vx + By * vy + Bz * vz
    E = U[4]

    expected_x = np.array(
        [
            rho * vx,
            rho * vx * vx + total_p - Bx**2,
            rho * vy * vx - Bx * By,
            rho * vz * vx - Bx * Bz,
            (E + total_p) * vx - Bx * Bdotv,
            0.0,
            vy * Bx - vx * By,
            vz * Bx - vx * Bz,
            0.0,
        ]
    )

    expected_y = np.array(
        [
            rho * vy,
            rho * vx * vy - By * Bx,
            rho * vy * vy + total_p - By**2,
            rho * vz * vy - By * Bz,
            (E + total_p) * vy - By * Bdotv,
            vx * By - vy * Bx,
            0.0,
            vz * By - vy * Bz,
            0.0,
        ]
    )

    expected_z = np.array(
        [
            rho * vz,
            rho * vx * vz - Bz * Bx,
            rho * vy * vz - Bz * By,
            rho * vz * vz + total_p - Bz**2,
            (E + total_p) * vz - Bz * Bdotv,
            vx * Bz - vz * Bx,
            vy * Bz - vz * By,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(model.flux_function(U, "x"), expected_x)
    assert np.allclose(model.flux_function(U, "y"), expected_y)
    assert np.allclose(model.flux_function(U, "z"), expected_z)


def test_source_terms():
    eta = 0.05
    mu_par, mu_perp = 1.0, 0.5
    kappa_par, kappa_perp = 2.0, 1.0
    model = ResistiveMHD(
        eta=eta,
        mu_parallel=mu_par,
        mu_perp=mu_perp,
        kappa_parallel=kappa_par,
        kappa_perp=kappa_perp,
        c_p=1.0,
    )

    primitives = np.array([1.0, 2.0, -1.0, 0.5, 0.5, 0.1, 0.2, 0.3])
    U = model.conservative_variables(primitives)
    grad_v = np.array([[0.1, 0.2, 0.3], [0.0, -0.1, 0.2], [0.3, 0.1, 0.0]])
    grad_T = np.array([0.5, -0.2, 0.1])

    src = model.source_terms(
        U, grad_v=grad_v, grad_T=grad_T, geometry="cylindrical", coord=2.0
    )

    rho, vx, vy, vz, p, Bx, By, Bz = primitives
    B = np.array([Bx, By, Bz])
    B2 = Bx**2 + By**2 + Bz**2
    expected = np.zeros_like(U)
    expected[4] += eta * B2
    expected[5:8] -= eta * B

    b_hat = B / (np.linalg.norm(B) + 1.0e-20)
    grad_para = np.outer(b_hat, grad_v @ b_hat)
    grad_perp = grad_v - grad_para
    visc_tensor = mu_par * grad_para + mu_perp * grad_perp
    force = np.sum(visc_tensor, axis=1)
    expected[1:4] += force
    v = np.array([vx, vy, vz])
    expected[4] += np.dot(force, v)

    grad_para_T = np.dot(grad_T, b_hat) * b_hat
    grad_perp_T = grad_T - grad_para_T
    heat_flux = -kappa_par * grad_para_T - kappa_perp * grad_perp_T
    expected[4] -= np.sum(heat_flux)

    r = 2.0
    p_th = model._pressure(U)
    expected[0] += -rho * vx / r
    expected[1] += (rho * (vy**2 + vz**2) + 0.5 * (By**2 + Bz**2) - Bx**2 + p_th) / r
    E = U[4]
    expected[4] += ((E + p_th + 0.5 * B2) * vx - Bx * (B @ v)) / r

    assert np.allclose(src, expected)


def test_shock_capturing_and_energy_closure():
    model = ResistiveMHD(c_h=1.0, c_p=1.0)
    left = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    right = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])

    U_left = model.conservative_variables(left)
    U_right = model.conservative_variables(right)
    U = np.vstack([U_left, U_right])
    dx = 1.0
    dt = 0.1 * dx / max(model.max_speed(U_left, "x"), model.max_speed(U_right, "x"))

    div0 = abs(U[1, 5] - U[0, 5])

    for _ in range(5):
        F_L = model.flux_function(U[0], "x")
        F_R = model.flux_function(U[1], "x")
        smax = max(model.max_speed(U[0], "x"), model.max_speed(U[1], "x"))
        flux = 0.5 * (F_L + F_R) - 0.5 * smax * (U[1] - U[0])
        U[0] -= dt / dx * flux
        U[1] += dt / dx * flux
        U[0] += dt * model.source_terms(U[0])
        U[1] += dt * model.source_terms(U[1])

    # Shock capturing: density on right cell increases
    assert U[1, 0] > right[0]

    # Energy closure
    for state in U:
        p = model._pressure(state)
        rho = state[0]
        v2 = np.sum((state[1:4] / rho) ** 2)
        B2 = np.sum(state[5:8] ** 2)
        E_calc = p / (model.gamma - 1.0) + 0.5 * rho * v2 + 0.5 * B2
        assert np.allclose(state[4], E_calc)

    # Divergence cleaning reduces magnetic divergence
    div1 = abs(U[1, 5] - U[0, 5])
    assert div1 < div0
