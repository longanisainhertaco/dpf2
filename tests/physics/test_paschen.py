import numpy as np

from dpf2.paschen import paschen_breakdown_time
from dpf2.prepulse import PrePulseBreakdownModel


def test_breakdown_delay_varies_across_gases():
    gap = 1.0
    voltage = 20.0
    pressures = {"He": 0.5, "Ar": 1.0, "Xe": 2.0}
    delays = {g: paschen_breakdown_time(gap, p, voltage) for g, p in pressures.items()}
    assert delays["He"] < delays["Ar"] < delays["Xe"]


def test_prepulse_uses_paschen_delay():
    time = np.linspace(0.0, 5.0, 11)
    current = np.linspace(0.0, 10.0, 11)
    model = PrePulseBreakdownModel(
        area=1.0,
        mass=1.0,
        force_threshold=1e6,
        gap=1.0,
        pressure=1.0,
        voltage=10.0,
    )
    res = model.run(time, current)
    t_paschen = paschen_breakdown_time(1.0, 1.0, 10.0)
    expected_idx = next((i for i, tt in enumerate(time) if tt >= t_paschen), len(time) - 1)
    assert res.breakdown_index == expected_idx
