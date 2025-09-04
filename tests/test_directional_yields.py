import json
import math
import pytest
import numpy as np

from dpf2.fusion import dd_beam_target_angular_spectrum, dd_directional_yields
from dpf2.synthetic_diagnostics import export_directional_yields


def test_directional_yields_symmetry(tmp_path):
    angles = [-180.0 + i for i in range(360)]
    spec = dd_beam_target_angular_spectrum(100.0, 1e18, 1e20, angles)
    totals = dd_directional_yields(100.0, 1e18, 1e20, bins=360)
    assert pytest.approx(sum(totals.values())) == float(sum(spec))
    assert totals["forward"] == pytest.approx(totals["backward"], rel=1e-3)
    out = export_directional_yields(tmp_path / "yields.json", totals)
    data = json.loads(out.read_text())
    assert data["forward"] == pytest.approx(totals["forward"])
