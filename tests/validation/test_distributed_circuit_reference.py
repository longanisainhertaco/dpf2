import json
from pathlib import Path
import numpy as np

from dpf2.circuit.distributed import (
    TransmissionLineSegment,
    TriggeredSwitch,
    assemble_matrices,
)


def test_assemble_matrices_reference():
    ref_path = Path(__file__).resolve().parents[2] / "ReferenceMaterial/distributed_circuit.json"
    data = json.loads(ref_path.read_text())
    seg = TransmissionLineSegment(
        from_node=0,
        to_node=1,
        length=1.0,
        L_per_m=1e-6,
        R_per_m=1.0,
        C_per_m=1e-6,
        L_parasitic=1e-9,
        R_parasitic=0.1,
        C_parasitic=2e-9,
    )
    sw = TriggeredSwitch(0, 1, closed=True, R_on=1e-3)
    R, L, C = assemble_matrices([seg], [sw])
    assert np.allclose(np.diag(R), np.array(data["R"]), rtol=1e-9)
    assert np.allclose(np.diag(L), np.array(data["L"]), rtol=1e-9)
    assert np.allclose(np.diag(C), np.array(data["C"]), rtol=1e-9)
