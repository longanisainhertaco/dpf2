import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "dpf2" / "physics"))
from mhd import ResistiveMHD


def test_conservative_variables():
    model = ResistiveMHD()
    primitives = np.array([1.0, 2.0, -1.0, 0.5, 0.1, 0.2, 0.3])
    U = model.conservative_variables(primitives)
    rho, v_r, v_z, p, B_r, B_z, B_phi = primitives
    kinetic = 0.5 * rho * (v_r**2 + v_z**2)
    magnetic = 0.5 * (B_r**2 + B_z**2 + B_phi**2)
    energy = p / (model.gamma - 1.0) + kinetic + magnetic
    expected = np.array([rho, rho*v_r, rho*v_z, energy, B_r, B_z, B_phi])
    assert np.allclose(U, expected)


def test_flux_function():
    model = ResistiveMHD()
    primitives = np.array([1.0, 2.0, -1.0, 0.5, 0.1, 0.2, 0.3])
    U = model.conservative_variables(primitives)
    rho, v_r, v_z, p, B_r, B_z, B_phi = primitives
    B2 = B_r**2 + B_z**2 + B_phi**2
    total_p = p + 0.5 * B2
    Bdotv = B_r * v_r + B_z * v_z
    E = U[3]
    expected_r = np.array([
        rho*v_r,
        rho*v_r*v_r + total_p - B_r**2,
        rho*v_z*v_r - B_r*B_z,
        (E + total_p)*v_r - B_r*Bdotv,
        0.0,
        v_z*B_r - v_r*B_z,
        B_phi*v_r,
    ])
    expected_z = np.array([
        rho*v_z,
        rho*v_r*v_z - B_r*B_z,
        rho*v_z*v_z + total_p - B_z**2,
        (E + total_p)*v_z - B_z*Bdotv,
        v_r*B_z - v_z*B_r,
        0.0,
        B_phi*v_z,
    ])
    assert np.allclose(model.flux_function(U, "r"), expected_r)
    assert np.allclose(model.flux_function(U, "z"), expected_z)


def test_source_terms():
    eta = 0.05
    model = ResistiveMHD(eta=eta)
    primitives = np.array([1.0, 2.0, -1.0, 0.5, 0.1, 0.2, 0.3])
    U = model.conservative_variables(primitives)
    B_r, B_z, B_phi = primitives[4:]
    B2 = B_r**2 + B_z**2 + B_phi**2
    expected = np.array([0.0, 0.0, 0.0, eta*B2, -eta*B_r, -eta*B_z, -eta*B_phi])
    assert np.allclose(model.source_terms(U), expected)
