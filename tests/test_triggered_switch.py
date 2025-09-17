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
    on_val = R_expected + sw.R_on
    off_val = R_expected + sw.R_off

    # Expect a 2x2 nodal matrix with values stamped onto the connected nodes
    expected_L = np.array([[L_expected, -L_expected], [-L_expected, L_expected]])
    expected_C = np.array([[C_expected, -C_expected], [-C_expected, C_expected]])
    expected_R_on = np.array([[on_val, -on_val], [-on_val, on_val]])
    expected_R_off = np.array([[off_val, -off_val], [-off_val, off_val]])

    assert np.allclose(L0, expected_L)
    assert np.allclose(C0, expected_C)
    assert np.allclose(R0, expected_R_on)
    assert np.allclose(R1, expected_R_off)


def test_branching_network_matrix_assembly():
    """Multiple segments and a switch form a small branched network."""

    seg1 = TransmissionLineSegment(
        0, 1, length=1.0, L_per_m=1.0, R_per_m=1.0, C_per_m=0.0
    )
    seg2 = TransmissionLineSegment(
        1, 2, length=1.0, L_per_m=0.0, R_per_m=2.0, C_per_m=0.0
    )
    sw = TriggeredSwitch(0, 2, closed=True, R_on=5.0, R_off=5.0)

    R, L, C = assemble_matrices([seg1, seg2], [sw], t=0.0)

    # Node order should be [0, 1, 2]
    R1 = seg1.totals()[1]
    R2 = seg2.totals()[1]
    R3 = sw.resistance(0.0)

    expected_R = np.array(
        [
            [R1 + R3, -R1, -R3],
            [-R1, R1 + R2, -R2],
            [-R3, -R2, R2 + R3],
        ]
    )

    L1 = seg1.totals()[0]
    expected_L = np.array(
        [
            [L1, -L1, 0.0],
            [-L1, L1, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert R.shape == (3, 3)
    assert np.allclose(R, expected_R)
    assert np.allclose(L, expected_L)
    assert np.allclose(C, np.zeros((3, 3)))
