import numpy as np
import pytest

from dpf2.circuit.distributed import (
    TransmissionLineSegment,
    TriggeredSwitch,
    ShuntCapacitance,
    StrayInductance,
    assemble_matrices,
)


def test_assemble_matrices_with_trigger_and_parasitics():
    seg = TransmissionLineSegment(
        length=1.0,
        L_per_m=2.0,
        R_per_m=3.0,
        C_per_m=4.0,
        parasitics=[ShuntCapacitance(1.0), StrayInductance(0.5)],
    )
    sw = TriggeredSwitch(trigger_time=1.0, R_on=0.1, R_off=10.0)

    R0, L0, C0 = assemble_matrices([seg], [sw], time=0.0)
    assert R0[0, 0] == 3.0 + 10.0
    assert L0[0, 0] == 2.0 + 0.5
    assert C0[0, 0] == 4.0 + 1.0

    R1, L1, C1 = assemble_matrices([seg], [sw], time=2.0)
    assert R1[0, 0] == 3.0 + 0.1
    assert L0[0, 0] == L1[0, 0]
    assert C0[0, 0] == C1[0, 0]
