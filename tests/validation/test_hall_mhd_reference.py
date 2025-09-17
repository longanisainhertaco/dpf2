import json
import math
from pathlib import Path
import numpy as np

from dpf2.physics import HallMHD


def test_snowplow_inductance_reference():
    ref_path = (
        Path(__file__).resolve().parents[2] / "ReferenceMaterial/hall_snowplow.json"
    )
    L_ref = json.loads(ref_path.read_text())["Lp"]
    model = HallMHD(current=1.0)
    B = math.sqrt(L_ref)
    prim = np.array([1.0, 0.0, 0.0, 0.0, 1.0, B, 0.0, 0.0])
    U = model.conservative_variables(prim)
    Lp = model.plasma_inductance(U)
    assert np.isclose(Lp, L_ref, rtol=0.05)
