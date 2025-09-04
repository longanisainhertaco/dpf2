#!/usr/bin/env python3
"""Train surrogate models for neutron yield and pinch time.

This utility loads the latest benchmark simulation data and fits a simple
linear regressor for each target.  The resulting models are exported to ONNX
and basic metadata including feature statistics and conformal calibration
quantiles are written to ``metadata.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - allow running without torch
    torch = None  # type: ignore
    nn = None  # type: ignore

# Paths -----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1] / "data" / "benchmarks"


def _peak_current(dataset: Path) -> float:
    """Return peak discharge current from ``current.csv``."""
    with (dataset / "current.csv").open() as fh:
        reader = csv.DictReader(fh)
        return max(float(row["value"]) for row in reader)


def _yield_and_pinch(dataset: Path) -> Tuple[float, float]:
    """Return final neutron yield and pinch time from ``neutron_yield.csv``."""
    with (dataset / "neutron_yield.csv").open() as fh:
        reader = csv.DictReader(fh)
        last = None
        for row in reader:
            last = row
    assert last is not None, f"no data in {dataset}/neutron_yield.csv"
    return float(last["value"]), float(last["time"])


def _prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load benchmark data and return feature and target arrays."""
    currents, yields, pinch_times = [], [], []
    for ds in DATA_ROOT.iterdir():
        if ds.is_dir():
            currents.append(_peak_current(ds))
            y, p = _yield_and_pinch(ds)
            yields.append(y)
            pinch_times.append(p)
    x = np.asarray(currents, dtype=np.float32).reshape(-1, 1)
    y = np.asarray(yields, dtype=np.float32).reshape(-1, 1)
    p = np.asarray(pinch_times, dtype=np.float32).reshape(-1, 1)
    return x, y, p


def _train_torch(x: np.ndarray, y: np.ndarray) -> "torch.nn.Module":
    """Fit a 1D linear regressor using PyTorch."""
    if torch is None:  # pragma: no cover - environment without torch
        raise RuntimeError("PyTorch is required for training")
    model = nn.Linear(1, 1)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x), torch.as_tensor(y)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(len(x), 32), shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    for _ in range(200):
        for xb, yb in loader:
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
    return model


def _conformal_quantile(model: "torch.nn.Module", x: np.ndarray, y: np.ndarray, q: float = 0.9) -> float:
    """Compute conformal calibration quantile of absolute residuals."""
    if torch is None:  # pragma: no cover - environment without torch
        raise RuntimeError("PyTorch is required for calibration")
    with torch.no_grad():
        preds = model(torch.as_tensor(x)).numpy()
    resid = np.abs(y - preds)
    return float(np.quantile(resid, q))


def _export_onnx(model: "torch.nn.Module", path: Path) -> None:
    """Export ``model`` to ONNX file at ``path``."""
    if torch is None:  # pragma: no cover - environment without torch
        raise RuntimeError("PyTorch is required for ONNX export")
    dummy = torch.zeros(1, 1)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["peak_current"],
        output_names=["prediction"],
        dynamic_axes={"peak_current": {0: "batch"}},
    )


def main() -> None:
    x, y, p = _prepare_data()
    feature_mean = float(x.mean())
    feature_var = float(x.var())

    if torch is None:
        raise RuntimeError("PyTorch is required to train surrogates")

    yield_model = _train_torch(x, y)
    pinch_model = _train_torch(x, p)

    q_y = _conformal_quantile(yield_model, x, y)
    q_p = _conformal_quantile(pinch_model, x, p)

    HERE.mkdir(parents=True, exist_ok=True)
    _export_onnx(yield_model, HERE / "yield_surrogate.onnx")
    _export_onnx(pinch_model, HERE / "pinch_time_surrogate.onnx")

    metadata = {
        "feature_mean": feature_mean,
        "feature_variance": feature_var,
        "yield": {"onnx": "yield_surrogate.onnx", "quantile": q_y},
        "pinch_time": {"onnx": "pinch_time_surrogate.onnx", "quantile": q_p},
    }
    with (HERE / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
