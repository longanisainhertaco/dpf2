"""Core driver for minimal DPF simulations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np

from dpf_config import DPFConfig
from circuit_config import CircuitConfig

from .circuit_solver import RLCCircuit, CircuitSolver
from .plasma_model import AnalyticPinchModel

__all__ = ["SimulationEngine"]


@dataclass
class SimulationResults:
    time: np.ndarray
    current: np.ndarray
    radius: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    neutron_yield: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "time": self.time.tolist(),
            "current": self.current.tolist(),
            "pinch_radius": self.radius.tolist(),
            "temperature": self.temperature.tolist(),
            "pressure": self.pressure.tolist(),
            "neutron_yield": self.neutron_yield,
        }


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
    def run(self, method: str = "analytical") -> SimulationResults:
        sc = self.config.simulation_control
        dt = sc.min_dt or 1e-9
        t_end = sc.time_end - sc.time_start
        circuit = self._setup_circuit()
        t, current = circuit.solve(t_end, dt, method=method)

        plasma = AnalyticPinchModel()
        pres = plasma.run(t, current)

        return SimulationResults(
            time=pres.time,
            current=current,
            radius=pres.radius,
            temperature=pres.temperature,
            pressure=pres.pressure,
            neutron_yield=pres.neutron_yield,
        )
