import json
from pathlib import Path
import numpy as np

from dpf2.physics import ResistiveMHD


def test_alegra_energy_profile():
    ref_path = (
        Path(__file__).resolve().parents[2] / "ReferenceMaterial/alegra_reference.json"
    )
    reference = json.loads(ref_path.read_text())

    model = ResistiveMHD(gamma=1.4)
    time = np.array(reference["time"])
    energies = []
    for t in time:
        rho = 1.0 - 0.4 * t
        p = 1.0 + t
        prim = np.array([rho, 0.0, 0.0, 0.0, p, 0.0, 0.0, 0.0])
        U = model.conservative_variables(prim)
        energies.append(U[4])

    atol = 0.0
    rtol = 1e-9
    assert np.allclose(time, reference["time"], rtol=rtol, atol=atol)
    assert np.allclose(np.array(energies), reference["energy"], rtol=rtol, atol=atol)
