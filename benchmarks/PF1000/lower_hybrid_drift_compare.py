from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dpf2.physics.lower_hybrid_drift import LowerHybridDrift

MU0 = 4e-7 * math.pi
PI = math.pi


def load_trace() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "Reference" / "PF1000" / "shot001.csv"
    times: list[float] = []
    currents: list[float] = []
    voltages: list[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_us"]))
            currents.append(float(row["current_norm"]))
            voltages.append(float(row["voltage_norm"]))
    return np.array(times), np.array(currents), np.array(voltages)


def predict_voltage(times: np.ndarray, current: np.ndarray, measured: np.ndarray) -> np.ndarray:
    radius = 0.01  # m
    n_i = 1e19
    k = 0.1
    amp = measured[0]
    preds = [amp]
    for i in range(1, len(times)):
        dt = (times[i] - times[i - 1]) * 1e-6
        B = MU0 * current[i] / (2 * PI * radius)
        model = LowerHybridDrift(B=B, n_i=n_i)
        amp = model.evolve(amp, k=k, dt=dt)
        preds.append(float(np.squeeze(amp)))
    return np.array(preds)


def flag_deviations(pred: np.ndarray, meas: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    dev = np.abs(pred - meas) / np.maximum(np.abs(meas), 1e-9)
    flagged = np.where(dev > threshold)[0]
    if flagged.size:
        print(f"lower_hybrid_drift: {flagged.size} points exceed {threshold*100:.0f}% deviation")
    else:
        print("lower_hybrid_drift: all points within threshold")
    return flagged


def main() -> None:
    times, current, voltage = load_trace()
    pred = predict_voltage(times, current, voltage)
    flagged = flag_deviations(pred, voltage)

    plt.figure(figsize=(6, 4))
    plt.plot(times, voltage, label="measured")
    plt.plot(times, pred, label="lhd prediction")
    if flagged.size:
        plt.scatter(times[flagged], pred[flagged], color="red", label=">20% dev")
    plt.xlabel("time (µs)")
    plt.ylabel("voltage (norm)")
    plt.legend()
    out = Path(__file__).with_name("lower_hybrid_drift_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
