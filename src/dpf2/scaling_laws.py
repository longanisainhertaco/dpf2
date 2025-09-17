from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Iterable, Sequence

import dataclasses
import numpy as np

from .simulation_engine import SimulationResults
from .core.config import DPFConfig
from .circuit_solver import RLCCircuit, CircuitSolver
from .pinch_models import AnalyticPinchModel


def _load_config(dataset_dir: Path) -> Dict[str, Any]:
    cfg_path = dataset_dir / "scaling.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            return json.load(f)
    return {}


def compare_to_scaling(
    results: SimulationResults, dataset_dir: Path
) -> Dict[str, float]:
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
        metrics["current_peak_error_pct"] = (
            abs(i_peak - i_pred) / i_pred * 100.0 if i_pred else float("inf")
        )

    if "neutron_yield_from_current" in cfg:
        law = cfg["neutron_yield_from_current"]
        k = float(law.get("k", 1.0))
        n = float(law.get("n", 1.0))
        y_pred = k * (i_peak**n)
        metrics["neutron_yield_pred"] = y_pred
        if y_pred:
            metrics["neutron_yield_error_pct"] = (
                abs(results.neutron_yield - y_pred) / y_pred * 100.0
            )
        else:
            metrics["neutron_yield_error_pct"] = float("inf")

    return metrics


def _fit_power_law(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Fit ``y = k * x**m`` returning ``(k, m)``.

    Only positive ``x`` and ``y`` entries are used in the fit.
    """

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = (x_arr > 0) & (y_arr > 0)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    logx = np.log(x_arr[mask])
    logy = np.log(y_arr[mask])
    m, logk = np.polyfit(logx, logy, 1)
    return float(np.exp(logk)), float(m)


def sweep_yield_scaling(
    base_config: DPFConfig,
    parameter: str,
    values: Iterable[float],
    *,
    t_end: float | None = None,
    dt: float | None = None,
) -> Dict[str, Any]:
    """Generate ``Y_n`` scaling data for a parameter sweep.

    The circuit is modelled using an analytic RLC solution and neutron yield is
    estimated with :class:`AnalyticPinchModel`.  The fitted power-law exponent
    ``m`` is returned for both ``Y_n`` vs. ``I_peak`` and ``Y_n`` vs. the swept
    parameter.
    """

    cfg = base_config
    model = AnalyticPinchModel()
    t_end = t_end or cfg.end_time
    dt = dt or t_end / 1000.0

    results: list[Dict[str, float]] = []

    if parameter == "initial_pressure":
        circuit = RLCCircuit(
            L=cfg.inductance,
            R=cfg.resistance,
            C=cfg.capacitance,
            V0=cfg.charging_voltage,
        )
        solver = CircuitSolver(circuit)
        t, base_current = solver.solve(t_end=t_end, dt=dt)
        base_p = cfg.initial_pressure
        for p in values:
            scale = (base_p / p) ** 0.5
            current = base_current * scale
            res = model.run(np.array(t), current)
            results.append(
                {
                    "parameter": float(p),
                    "I_peak": float(np.max(current)),
                    "yield": float(res.neutron_yield),
                }
            )
    else:
        for val in values:
            new_cfg = dataclasses.replace(cfg, **{parameter: val})
            circuit = RLCCircuit(
                L=new_cfg.inductance,
                R=new_cfg.resistance,
                C=new_cfg.capacitance,
                V0=new_cfg.charging_voltage,
            )
            solver = CircuitSolver(circuit)
            t, current = solver.solve(t_end=t_end, dt=dt)
            res = model.run(np.array(t), current)
            results.append(
                {
                    "parameter": float(val),
                    "I_peak": float(np.max(current)),
                    "yield": float(res.neutron_yield),
                }
            )

    params = [r["parameter"] for r in results]
    i_peaks = [r["I_peak"] for r in results]
    yields = [r["yield"] for r in results]
    _, m_current = _fit_power_law(i_peaks, yields)
    _, m_param = _fit_power_law(params, yields)
    return {
        "parameter": parameter,
        "values": params,
        "I_peak": i_peaks,
        "Y_n": yields,
        "m_current": m_current,
        "m_parameter": m_param,
    }


__all__ = ["compare_to_scaling", "sweep_yield_scaling"]
