import numpy as np

from dpf2.physics import ResistiveMHD
from dpf2.mesh import Mesh3D


def _riemann_function(p, rho, p_k, a, gamma):
    if p > p_k:  # shock
        A = 2.0 / ((gamma + 1.0) * rho)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        f = (p - p_k) * np.sqrt(A / (p + B))
        df = np.sqrt(A / (p + B)) * (1.0 - 0.5 * (p - p_k) / (p + B))
    else:  # rarefaction
        f = (
            2.0
            * a
            / (gamma - 1.0)
            * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        )
        df = (1.0 / (rho * a)) * (p / p_k) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df


def _sod_exact(x, t, left, right, gamma):
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    p = 0.5 * (p_l + p_r)
    for _ in range(20):
        f_l, df_l = _riemann_function(p, rho_l, p_l, a_l, gamma)
        f_r, df_r = _riemann_function(p, rho_r, p_r, a_r, gamma)
        p_new = p - (f_l + f_r + u_r - u_l) / (df_l + df_r)
        if abs(p_new - p) / max(p_new, p, 1.0) < 1e-6:
            p = p_new
            break
        p = p_new
    p_star = p
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)
    if p_star > p_l:
        rho_star_l = rho_l * (
            (p_star / p_l + (gamma - 1.0) / (gamma + 1.0))
            / ((gamma - 1.0) / (gamma + 1.0) * p_star / p_l + 1.0)
        )
        S_L = u_l - a_l * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * p_star / p_l + (gamma - 1.0) / (2.0 * gamma)
        )
    else:
        rho_star_l = rho_l * (p_star / p_l) ** (1.0 / gamma)
        a_star_l = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
        S_HL = u_l - a_l
        S_TL = u_star - a_star_l
    if p_star > p_r:
        rho_star_r = rho_r * (
            (p_star / p_r + (gamma - 1.0) / (gamma + 1.0))
            / ((gamma - 1.0) / (gamma + 1.0) * p_star / p_r + 1.0)
        )
        S_R = u_r + a_r * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * p_star / p_r + (gamma - 1.0) / (2.0 * gamma)
        )
    else:
        rho_star_r = rho_r * (p_star / p_r) ** (1.0 / gamma)
        a_star_r = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
        S_HR = u_r + a_r
        S_TR = u_star + a_star_r
    xi = (x - 0.5) / t
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p_out = np.zeros_like(x)
    for i, s in enumerate(xi):
        if s < u_star:
            if p_star > p_l:  # left shock
                if s < S_L:
                    rho[i], u[i], p_out[i] = rho_l, u_l, p_l
                else:
                    rho[i], u[i], p_out[i] = rho_star_l, u_star, p_star
            else:  # left rarefaction
                if s < S_HL:
                    rho[i], u[i], p_out[i] = rho_l, u_l, p_l
                elif s > S_TL:
                    rho[i], u[i], p_out[i] = rho_star_l, u_star, p_star
                else:
                    u[i] = (2.0 / (gamma + 1.0)) * (a_l + s)
                    a = a_l + 0.5 * (gamma - 1.0) * (u[i] - u_l)
                    rho[i] = rho_l * (a / a_l) ** (2.0 / (gamma - 1.0))
                    p_out[i] = p_l * (a / a_l) ** (2.0 * gamma / (gamma - 1.0))
        else:
            if p_star > p_r:  # right shock
                if s > S_R:
                    rho[i], u[i], p_out[i] = rho_r, u_r, p_r
                else:
                    rho[i], u[i], p_out[i] = rho_star_r, u_star, p_star
            else:  # right rarefaction
                if s > S_HR:
                    rho[i], u[i], p_out[i] = rho_r, u_r, p_r
                elif s < S_TR:
                    rho[i], u[i], p_out[i] = rho_star_r, u_star, p_star
                else:
                    u[i] = (2.0 / (gamma + 1.0)) * (-a_r + s)
                    a = a_r - 0.5 * (gamma - 1.0) * (u[i] - u_r)
                    rho[i] = rho_r * (a / a_r) ** (2.0 / (gamma - 1.0))
                    p_out[i] = p_r * (a / a_r) ** (2.0 * gamma / (gamma - 1.0))
    return rho, u, p_out


def test_sod_shock_tube_regression():
    gamma = 1.4
    model = ResistiveMHD(gamma=gamma)
    n = 400
    mesh = Mesh3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n, 1, 1)
    left = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    U = np.zeros((n, len(model.equations)))
    for i in range(n):
        prim = left if i < n // 2 else right
        U[i] = model.conservative_variables(prim)
    dx = mesh.dx
    t = 0.0
    t_end = 0.1
    while t < t_end:
        max_speeds = [model.max_speed(state, "x") for state in U]
        dt = 0.4 * dx / max(max_speeds)
        if t + dt > t_end:
            dt = t_end - t
        fluxes = np.zeros((n + 1, len(model.equations)))
        for i in range(n - 1):
            F_L = model.flux_function(U[i], "x")
            F_R = model.flux_function(U[i + 1], "x")
            smax = max(model.max_speed(U[i], "x"), model.max_speed(U[i + 1], "x"))
            fluxes[i + 1] = 0.5 * (F_L + F_R) - 0.5 * smax * (U[i + 1] - U[i])
        fluxes[0] = model.flux_function(U[0], "x")
        fluxes[-1] = model.flux_function(U[-1], "x")
        for i in range(n):
            U[i] -= dt / dx * (fluxes[i + 1] - fluxes[i])
        t += dt
    x = 0.5 * (mesh.x[:-1] + mesh.x[1:])
    rho_exact, u_exact, p_exact = _sod_exact(
        x, t_end, (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), gamma
    )
    rho_num = U[:, 0]
    diff = np.abs(rho_num - rho_exact)
    l1 = sum(diff.data) / len(diff)
    assert l1 < 0.1
