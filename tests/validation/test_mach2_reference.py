import json
from pathlib import Path
import numpy as np

from dpf2.physics import ResistiveMHD


def test_mach2_flux_reference():
    ref_path = Path(__file__).resolve().parents[2] / "ReferenceMaterial/mach2_reference.json"
    reference = json.loads(ref_path.read_text())

    model = ResistiveMHD(gamma=1.4)
    prim = np.array([1.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    U = model.conservative_variables(prim)
    flux = model.flux_function(U, "x")

    atol = 0.0
    rtol = 1e-9
    assert np.allclose(flux, reference["flux_x"], rtol=rtol, atol=atol)
