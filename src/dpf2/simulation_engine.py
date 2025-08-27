"""Core driver for minimal DPF simulations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .dpf_config import DPFConfig
from .circuit_config import CircuitConfig

from .circuit_solver import RLCCircuit, CircuitSolver
from .pinch_models import (
    AnalyticPinchModel,
    SemiAnalyticPinchModel,
    PinchModelBase,
    MHDPinchModel,
)
from .experimental_variability import MonteCarloVariability

__all__ = ["SimulationEngine", "SimulationResults", "EnsembleResults"]


@dataclass
class SimulationResults:
    time: np.ndarray
    current: np.ndarray
    radius: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    neutron_yield: float
    axial_position: np.ndarray | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "time": self.time.tolist(),
            "current": self.current.tolist(),
            "pinch_radius": self.radius.tolist(),
            "temperature": self.temperature.tolist(),
            "pressure": self.pressure.tolist(),
            "neutron_yield": self.neutron_yield,
            **({"axial_position": self.axial_position.tolist()} if self.axial_position is not None else {}),
        }


@dataclass
class EnsembleResults:
    """Statistics aggregated from multiple realizations."""

    time: np.ndarray
    current_mean: np.ndarray
    current_std: np.ndarray
    radius_mean: np.ndarray
    radius_std: np.ndarray
    temperature_mean: np.ndarray
    temperature_std: np.ndarray
    pressure_mean: np.ndarray
    pressure_std: np.ndarray
    neutron_yield_mean: float
    neutron_yield_std: float
    axial_position_mean: Optional[np.ndarray] | None = None
    axial_position_std: Optional[np.ndarray] | None = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "time": self.time.tolist(),
            "current_mean": self.current_mean.tolist(),
            "current_std": self.current_std.tolist(),
            "pinch_radius_mean": self.radius_mean.tolist(),
            "pinch_radius_std": self.radius_std.tolist(),
            "temperature_mean": self.temperature_mean.tolist(),
            "temperature_std": self.temperature_std.tolist(),
            "pressure_mean": self.pressure_mean.tolist(),
            "pressure_std": self.pressure_std.tolist(),
            "neutron_yield_mean": self.neutron_yield_mean,
            "neutron_yield_std": self.neutron_yield_std,
        }
        if self.axial_position_mean is not None and self.axial_position_std is not None:
            data.update(
                {
                    "axial_position_mean": self.axial_position_mean.tolist(),
                    "axial_position_std": self.axial_position_std.tolist(),
                }
            )
        return data


class SimulationEngine:
    """Execute a minimal Dense Plasma Focus simulation."""

    def __init__(self, config: DPFConfig) -> None:
        self.config = config.resolve_defaults()

    # ------------------------------------------------------------------
    def _setup_circuit(self) -> CircuitSolver:
        cc: CircuitConfig = self.config.circuit_config
        circuit = RLCCircuit(
            L=cc.L_ext * 1e-6,
            R=cc.R_ext * 1e-3,
            C=cc.C_ext * 1e-6,
            V0=cc.V0 * 1e3,
        )
        return CircuitSolver(circuit)

    # ------------------------------------------------------------------
    def run(
        self,
        method: str = "analytical",
        pinch_model: str = "analytic",
        variability: Optional[MonteCarloVariability] = None,
    ) -> SimulationResults | EnsembleResults:
        if variability is None:
            sc = self.config.simulation_control
            dt = sc.min_dt or 1e-9
            t_end = sc.time_end - sc.time_start
            circuit = self._setup_circuit()
            t, current = circuit.solve(t_end, dt, method=method)

            if pinch_model == "analytic":
                plasma: PinchModelBase = AnalyticPinchModel()
            elif pinch_model == "semi-analytic":
                plasma = SemiAnalyticPinchModel()
            elif pinch_model == "mhd":
                plasma = MHDPinchModel()
            else:
                raise ValueError(
                    "pinch_model must be 'analytic', 'semi-analytic', or 'mhd'",
                )
            pres = plasma.run(t, current)

            return SimulationResults(
                time=pres.time,
                current=current,
                radius=pres.radius,
                temperature=pres.temperature,
                pressure=pres.pressure,
                neutron_yield=pres.neutron_yield,
                axial_position=pres.axial_position,
            )

        configs = variability.generate_configurations(self.config)
        results: List[SimulationResults] = []
        for cfg in configs:
            engine = SimulationEngine(cfg)
            results.append(engine.run(method=method, pinch_model=pinch_model))

        time = results[0].time
        stack = lambda attr: np.vstack([getattr(r, attr) for r in results])
        current = stack("current")
        radius = stack("radius")
        temp = stack("temperature")
        pres = stack("pressure")

        axial_mean: Optional[np.ndarray] = None
        axial_std: Optional[np.ndarray] = None
        if all(r.axial_position is not None for r in results):
            ax = stack("axial_position")
            axial_mean = ax.mean(axis=0)
            axial_std = ax.std(axis=0)

        yields = np.array([r.neutron_yield for r in results])

        return EnsembleResults(
            time=time,
            current_mean=current.mean(axis=0),
            current_std=current.std(axis=0),
            radius_mean=radius.mean(axis=0),
            radius_std=radius.std(axis=0),
            temperature_mean=temp.mean(axis=0),
            temperature_std=temp.std(axis=0),
            pressure_mean=pres.mean(axis=0),
            pressure_std=pres.std(axis=0),
            neutron_yield_mean=float(yields.mean()),
            neutron_yield_std=float(yields.std()),
            axial_position_mean=axial_mean,
            axial_position_std=axial_std,
        )

