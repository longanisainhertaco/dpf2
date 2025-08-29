import json
from pathlib import Path

import numpy as np

from dpf2.physics import ResistiveMHD
from dpf2.circuit_config import CircuitConfig
from dpf2.circuit_solver import run_circuit_simulation


def test_cross_code_and_experiment_regression():
    ref_dir = Path(__file__).resolve().parents[2] / "ReferenceMaterial"

    # ALEGRA reference energy deposition
    alegra = json.loads((ref_dir / "alegra_reference.json").read_text())
    model = ResistiveMHD(gamma=1.4)
    time = np.array(alegra["time"])
    energies = []
    for t in time:
        rho = 1.0 - 0.4 * t
        p = 1.0 + t
        prim = np.array([rho, 0.0, 0.0, 0.0, p, 0.0, 0.0, 0.0])
        U = model.conservative_variables(prim)
        energies.append(U[4])
    assert np.allclose(np.array(energies), alegra["energy"], rtol=1e-9, atol=0.0)

    # MACH2 flux reference
    mach2 = json.loads((ref_dir / "mach2_reference.json").read_text())
    prim = np.array([1.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    U = model.conservative_variables(prim)
    flux = model.flux_function(U, "x")
    assert np.allclose(flux, mach2["flux_x"], rtol=1e-9, atol=0.0)

    # High-resolution experimental shot
    exp = json.loads((ref_dir / "experimental_shot_highres.json").read_text())
    cc = CircuitConfig(L_ext=1.0, R_ext=2.0, C_ext=0.5, V0=1.0, switch_delay=0.0)
    num = len(exp["time"])
    t, current, voltage, _, _ = run_circuit_simulation(cc, t_end=1.0, num_points=num)
    pressure = [0.01 * i**2 for i in current]
    temperature = [300 + 0.001 * i for i in current]
    neutron_yield = sum(pressure) * 1e-6

    atol = 0.0
    rtol = 1e-9
    assert np.allclose(t, exp["time"], rtol=rtol, atol=atol)
    assert np.allclose(current, exp["current"], rtol=rtol, atol=atol)
    assert np.allclose(voltage, exp["voltage"], rtol=rtol, atol=atol)
    assert np.allclose(pressure, exp["pressure"], rtol=rtol, atol=atol)
    assert np.allclose(temperature, exp["temperature"], rtol=rtol, atol=atol)
    assert np.isclose(neutron_yield, exp["neutron_yield"], rtol=rtol, atol=atol)
