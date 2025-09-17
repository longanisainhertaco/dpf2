# Benchmark against UNU/ICTP Plasma Focus Facility waveform data.
#
# This benchmark loads a simple current waveform and runs the coupled
# pre-pulse, sheath, and pinch simulation. The metric reported is the
# difference between the measured peak current time and the simulated
# pinch time.

from __future__ import annotations

import numpy as np

from dpf2.prepulse import PrePulseBreakdownModel
from dpf2.axial_sheath import AxialSheathModel
from dpf2.pinch_models import PinchModelBase, PinchResult
from dpf2.coupled_models import CoupledEndToEndModel


def _load_waveform() -> tuple[np.ndarray, np.ndarray]:
    time_vals = []
    current_vals = []
    with open("Reference/UNU/shot001.csv") as fh:
        next(fh)
        for line in fh:
            t_str, i_str, _ = line.strip().split(",")
            time_vals.append(float(t_str))
            current_vals.append(float(i_str))
    time = np.array(time_vals) * 1e-6
    current = np.array(current_vals) * 1e4
    return time, current


def run_benchmark() -> dict[str, float]:
    time, current = _load_waveform()
    pre = PrePulseBreakdownModel(area=1.0, mass=1.0, force_threshold=1e-5)
    sheath = AxialSheathModel(area=1.0, mass=1.0, length=0.1)

    class _MiniPinch(PinchModelBase):
        def run(self, time, current):
            t = np.array(list(time))
            zeros = np.array([0.0 for _ in t])
            return PinchResult(
                time=t,
                radius=zeros,
                temperature=zeros,
                pressure=zeros,
                neutron_yield=0.0,
            )

    pinch = _MiniPinch()
    model = CoupledEndToEndModel(pre, sheath, pinch)
    result = model.run(time, current)
    peak_idx = max(range(len(current)), key=lambda i: current[i])
    peak_t = time[peak_idx]
    if len(result.pinch.time):
        min_idx = max(
            range(len(result.pinch.radius)), key=lambda i: -result.pinch.radius[i]
        )
        pinch_t = result.pinch.time[min_idx]
    else:
        pinch_t = time[-1]
    return {"pinch_time_error": float(abs(pinch_t - peak_t))}


if __name__ == "__main__":
    metrics = run_benchmark()
    for k, v in metrics.items():
        print(f"{k}: {v:.3e}")
