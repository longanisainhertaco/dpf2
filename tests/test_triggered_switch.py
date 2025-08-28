import numpy as np

from dpf2.circuit.distributed import (
    TransmissionLineSegment,
    TriggeredSwitch,
    assemble_matrices,
)


def test_triggered_switch_and_matrix_assembly():
    seg = TransmissionLineSegment(
        from_node=0,
        to_node=1,
        length=2.0,
        L_per_m=1.0,
        R_per_m=2.0,
        C_per_m=3.0,
        L_parasitic=0.1,
        R_parasitic=0.2,
        C_parasitic=0.3,
    )
    sw = TriggeredSwitch(
        from_node=0,
        to_node=1,
        closed=True,
        R_on=1.0,
        R_off=10.0,
        trigger_time=1e-6,
    )

    R0, L0, C0 = assemble_matrices([seg], [sw], t=0.0)
    R1, _, _ = assemble_matrices([seg], [sw], t=2e-6)

    L_expected, R_expected, C_expected = seg.totals()
    assert np.allclose(L0[0, 0], L_expected)
    assert np.allclose(C0[0, 0], C_expected)
    assert np.allclose(R0[0, 0], R_expected + sw.R_on)
    assert np.allclose(R1[0, 0], R_expected + sw.R_off)

