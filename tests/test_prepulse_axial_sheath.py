import numpy as np

import numpy as np

from dpf2.prepulse import PrePulseBreakdownModel, mu0
from dpf2.axial_sheath import AxialSheathModel
from dpf2.coupled_models import CoupledEndToEndModel
from dpf2.pinch_models import PinchModelBase, PinchResult


def test_prepulse_breakdown_detects_threshold():
    time = np.array([0.0, 1.0, 2.0])
    current = np.array([0.0, 10.0, 20.0])
    model = PrePulseBreakdownModel(area=1.0, mass=1.0, force_threshold=1e-4)
    res = model.run(time, current)
    assert res.breakdown_index == 2
    radius = np.sqrt(1 / np.pi)
    expected = mu0 / (2 * np.pi * radius) * current * current
    assert np.allclose(res.jxb_force, expected)


def test_axial_sheath_advances():
    time = np.array([0.0, 1.0, 2.0])
    current = np.array([0.0, 10.0, 20.0])
    sheath = AxialSheathModel(area=1.0, mass=1.0, length=0.1)
    res = sheath.run(time, current)
    assert res.position[-1] > res.position[0]


class DummyPinch(PinchModelBase):
    def run(self, time, current):
        t = np.array(list(time))
        zeros = np.array([0.0 for _ in t])
        return PinchResult(
            time=t, radius=zeros, temperature=zeros, pressure=zeros, neutron_yield=0.0
        )


def test_coupled_model_runs():
    time = np.array([0.0, 1.0, 2.0])
    current = np.array([0.0, 10.0, 20.0])
    pre = PrePulseBreakdownModel(area=1.0, mass=1.0, force_threshold=1e-5)
    sheath = AxialSheathModel(area=1.0, mass=1.0, length=0.1)
    pinch = DummyPinch()
    model = CoupledEndToEndModel(pre, sheath, pinch)
    result = model.run(time, current)
    assert len(result.pinch.time) == len(time) - result.sheath.end_index
