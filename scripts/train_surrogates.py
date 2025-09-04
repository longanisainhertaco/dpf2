#!/usr/bin/env python3
"""Train simple linear surrogates for yield and pinch time.

The training data is taken from the high-fidelity benchmark runs located
under ``data/benchmarks``. For each benchmark the peak discharge current is
used as the single input feature while neutron yield and pinch time are the
prediction targets.  The resulting models are stored in ``ai/surrogates`` as
JSON files containing both the regression coefficients and metadata such as
the training domain and mean absolute error.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Tuple, List


def _peak_current(dataset: Path) -> float:
    with (dataset / "current.csv").open() as fh:
        reader = csv.DictReader(fh)
        return max(float(row["value"]) for row in reader)


def _yield_and_pinch(dataset: Path) -> Tuple[float, float]:
    with (dataset / "neutron_yield.csv").open() as fh:
        reader = csv.DictReader(fh)
        last = None
        for row in reader:
            last = row
    assert last is not None, f"no data in {dataset}/neutron_yield.csv"
    return float(last["value"]), float(last["time"])


def _linear_regression(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    x_list = list(x)
    y_list = list(y)
    n = len(x_list)
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n
    s_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_list, y_list))
    s_xx = sum((xi - mean_x) ** 2 for xi in x_list)
    slope = s_xy / s_xx if s_xx else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_t = list(y_true)
    y_p = list(y_pred)
    n = len(y_t)
    return sum(abs(a - b) for a, b in zip(y_t, y_p)) / n


def train() -> None:
    data_dir = Path("data/benchmarks")
    currents: List[float] = []
    yields: List[float] = []
    pinch_times: List[float] = []
    for ds in data_dir.iterdir():
        if ds.is_dir():
            currents.append(_peak_current(ds))
            y, p = _yield_and_pinch(ds)
            yields.append(y)
            pinch_times.append(p)

    domain = [min(currents), max(currents)]
    mean = sum(currents) / len(currents)
    var = sum((c - mean) ** 2 for c in currents) / len(currents)

    # Two-sigma Mahalanobis threshold ~95% for Gaussian data
    ood_threshold = 2.0

    # Yield model -------------------------------------------------------
    a_y, b_y = _linear_regression(currents, yields)
    y_pred = [a_y * c + b_y for c in currents]
    err_y = _mae(yields, y_pred)

    out_dir = Path("ai/surrogates")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "yield_model.json").open("w") as fh:
        json.dump(
            {
                "coeffs": [a_y, b_y],
                "training_domain": domain,
                "error": err_y,
                "mean": mean,
                "covariance": var,
                "ood_threshold": ood_threshold,
            },
            fh,
            indent=2,
        )

    # Pinch time model --------------------------------------------------
    a_p, b_p = _linear_regression(currents, pinch_times)
    p_pred = [a_p * c + b_p for c in currents]
    err_p = _mae(pinch_times, p_pred)

    with (out_dir / "pinch_time_model.json").open("w") as fh:
        json.dump(
            {
                "coeffs": [a_p, b_p],
                "training_domain": domain,
                "error": err_p,
                "mean": mean,
                "covariance": var,
                "ood_threshold": ood_threshold,
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    train()
