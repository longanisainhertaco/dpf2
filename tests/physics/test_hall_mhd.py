import numpy as np

from dpf2.physics import HallMHD


def _shock_setup():
    model = HallMHD(hall_coeff=0.1)
    left = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    right = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    U_left = model.conservative_variables(left)
    U_right = model.conservative_variables(right)
    return model, U_left, U_right


def test_shock_propagation():
    model, U_left, U_right = _shock_setup()
    U = np.vstack([U_left, U_right])
    dx = 1.0
    dt = 0.1 * dx / max(model.max_speed(U_left, "x"), model.max_speed(U_right, "x"))

    for _ in range(5):
        F_L = model.flux_function(U[0], "x")
        F_R = model.flux_function(U[1], "x")
        smax = max(model.max_speed(U[0], "x"), model.max_speed(U[1], "x"))
        flux = 0.5 * (F_L + F_R) - 0.5 * smax * (U[1] - U[0])
        U[0] -= dt / dx * flux
        U[1] += dt / dx * flux

    assert U[1, 0] > U_right[0]
    for state in U:
        prim = model.primitive_variables(state)
        cons = model.conservative_variables(prim)
        assert np.allclose(cons, state)


def test_alfven_wave_propagation():
    model = HallMHD()
    n = 32
    x = np.arange(n)
    rho0 = 1.0
    B0 = 1.0
    amp = 1.0e-2
    By = amp * np.sin(2 * np.pi * x / n)
    vy = -amp * np.sin(2 * np.pi * x / n)

    U = np.zeros((n, 9))
    for i in range(n):
        prim = np.array([rho0, 0.0, vy[i], 0.0, 1.0, 0.0, By[i], B0])
        U[i] = model.conservative_variables(prim)

    dx = 1.0
    ca = B0 / np.sqrt(rho0)
    dt = 0.4 * dx / ca

    for _ in range(5):
        By = U[:, 6]
        Jz = np.gradient(By, dx)
        J = np.zeros((n, 3))
        J[:, 2] = Jz
        fluxes = np.zeros((n + 1, 9))
        for i in range(n):
            UL = U[i]
            UR = U[(i + 1) % n]
            F_L = model.flux_function(UL, "x", J=J[i])
            F_R = model.flux_function(UR, "x", J=J[(i + 1) % n])
            smax = max(model.max_speed(UL, "x"), model.max_speed(UR, "x"))
            fluxes[i + 1] = 0.5 * (F_L + F_R) - 0.5 * smax * (UR - UL)
        fluxes[0] = fluxes[n]
        for i in range(n):
            U[i] -= dt / dx * (fluxes[i + 1] - fluxes[i])

    final_By = U[:, 6]
    assert np.isclose(np.max(np.abs(final_By)), amp, rtol=0.2)
