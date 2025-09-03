from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .simulation_engine import SimulationResults


def _load_config(dataset_dir: Path) -> Dict[str, Any]:
    cfg_path = dataset_dir / "scaling.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            return json.load(f)
    return {}


def compare_to_scaling(results: SimulationResults, dataset_dir: Path) -> Dict[str, float]:
    """Compare simulation outputs to simple scaling law predictions.

    Parameters
    ----------
    results:
        Simulation results from :class:`SimulationEngine`.
    dataset_dir:
        Directory containing ``scaling.json`` parameters.

    Returns
    -------
    dict
        Dictionary of discrepancy metrics. Empty if no scaling info.
    """
    cfg = _load_config(dataset_dir)
    if not cfg:
        return {}

    metrics: Dict[str, float] = {}

    # Peak values in convenient units
    i_peak = float(np.max(results.current) / 1e3)  # kA
    v_peak = float(np.max(results.voltage) / 1e3) if results.voltage.size else 0.0  # kV
    metrics["I_peak_kA"] = i_peak
    metrics["V_peak_kV"] = v_peak

    if "current_peak_from_voltage" in cfg and v_peak:
        law = cfg["current_peak_from_voltage"]
        k = float(law.get("k", 1.0))
        n = float(law.get("n", 1.0))
        i_pred = k * v_peak**n
        metrics["current_pred_kA"] = i_pred
        metrics["current_peak_error_pct"] = abs(i_peak - i_pred) / i_pred * 100.0 if i_pred else float("inf")

    if "neutron_yield_from_current" in cfg:
        law = cfg["neutron_yield_from_current"]
        k = float(law.get("k", 1.0))
        n = float(law.get("n", 1.0))
        y_pred = k * (i_peak**n)
        metrics["neutron_yield_pred"] = y_pred
        if y_pred:
            metrics["neutron_yield_error_pct"] = abs(results.neutron_yield - y_pred) / y_pred * 100.0
        else:
            metrics["neutron_yield_error_pct"] = float("inf")

    return metrics

__all__ = ["compare_to_scaling"]
