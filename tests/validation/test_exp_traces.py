import json
from pathlib import Path

import numpy as np
from dpf2.circuit_config import CircuitConfig
from dpf2.circuit_solver import run_circuit_simulation


def test_z_machine_traces():
    ref_path = Path(__file__).resolve().parents[2] / "ReferenceMaterial/z_machine_traces.json"
    reference = json.loads(ref_path.read_text())

    cc = CircuitConfig(L_ext=1.0, R_ext=1.0, C_ext=1.0, V0=1.0, switch_delay=0.0)
    num = len(reference["time"])
    t, current, voltage, _, _ = run_circuit_simulation(cc, t_end=1.0, num_points=num)

    pressure = [1e-2 * i**2 for i in current]
    temperature = [300 + 1e-3 * i for i in current]
    neutron_yield = sum(pressure) * 1e-6

    atol = 0.0
    rtol = 1e-9

    assert np.allclose(t, reference["time"], rtol=rtol, atol=atol)
    assert np.allclose(current, reference["current"], rtol=rtol, atol=atol)
    assert np.allclose(voltage, reference["voltage"], rtol=rtol, atol=atol)
    assert np.allclose(pressure, reference["pressure"], rtol=rtol, atol=atol)
    assert np.allclose(temperature, reference["temperature"], rtol=rtol, atol=atol)
    assert np.isclose(neutron_yield, reference["neutron_yield"], rtol=rtol, atol=atol)
