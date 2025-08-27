from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .dpf_config import DPFConfig
from .circuit_config import CircuitConfig

from .circuit_solver import RLCCircuit, CircuitSolver, run_circuit_simulation
from .core.bases import PlasmaSolverBase
from .pinch_models import (
    AnalyticPinchModel,
    SemiAnalyticPinchModel,
    PinchModelBase,
    MHDPinchModel,
)

__all__ = ["SimulationEngine"]


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
        plasma_solver: PlasmaSolverBase | None = None,
    ) -> SimulationResults:
        sc = self.config.simulation_control
        dt = sc.min_dt or 1e-9
        t_end = sc.time_end - sc.time_start

        if plasma_solver is not None:
            cc: CircuitConfig = self.config.circuit_config
            num = int(t_end / dt) + 1
            t, current, _, _, _ = run_circuit_simulation(
                cc, t_end * 1e6, num_points=num, plasma_solver=plasma_solver, plasma_state=None
            )
            zeros = np.zeros_like(t)
            return SimulationResults(
                time=t,
                current=current,
                radius=zeros,
                temperature=zeros,
                pressure=zeros,
                neutron_yield=0.0,
                axial_position=None,
            )

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
                "pinch_model must be 'analytic', 'semi-analytic', or 'mhd'"
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
