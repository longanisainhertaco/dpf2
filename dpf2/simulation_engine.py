"""Core driver for minimal DPF simulations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from dpf_config import DPFConfig
from circuit_config import CircuitConfig

from .circuit_solver import RLCCircuit, CircuitSolver
from .pinch_models import AnalyticPinchModel, SemiAnalyticPinchModel, PinchModelBase
from .radiation import K_B, total_radiation_loss

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
    radiation_losses: Optional[Dict[str, np.ndarray]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "time": self.time.tolist(),
            "current": self.current.tolist(),
            "pinch_radius": self.radius.tolist(),
            "temperature": self.temperature.tolist(),
            "pressure": self.pressure.tolist(),
            "neutron_yield": self.neutron_yield,
            **(
                {"axial_position": self.axial_position.tolist()}
                if self.axial_position is not None
                else {}
            ),
            **(
                {"radiation_losses": {k: v.tolist() for k, v in self.radiation_losses.items()}}
                if self.radiation_losses is not None
                else {}
            ),
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
    def run(self, method: str = "analytical", pinch_model: str = "analytic") -> SimulationResults:
        sc = self.config.simulation_control
        dt = sc.min_dt or 1e-9
        t_end = sc.time_end - sc.time_start
        circuit = self._setup_circuit()
        t, current = circuit.solve(t_end, dt, method=method)

        if pinch_model == "analytic":
            plasma: PinchModelBase = AnalyticPinchModel()
        elif pinch_model == "semi-analytic":
            plasma = SemiAnalyticPinchModel()
        else:
            raise ValueError("pinch_model must be 'analytic' or 'semi-analytic'")
        pres = plasma.run(t, current)

        zeff = getattr(plasma, "zeff", 1.0)
        n_i = pres.pressure / ((zeff + 1.0) * K_B * pres.temperature)
        n_e = zeff * n_i
        losses = total_radiation_loss(pres.temperature, n_e, n_i, zeff)

        dt_arr = np.diff(pres.time, prepend=pres.time[0])
        dT = losses["total"] * dt_arr / (1.5 * (n_e + n_i) * K_B)
        temperature = np.maximum(pres.temperature - np.cumsum(dT), 0.0)

        return SimulationResults(
            time=pres.time,
            current=current,
            radius=pres.radius,
            temperature=temperature,
            pressure=pres.pressure,
            neutron_yield=pres.neutron_yield,
            axial_position=pres.axial_position,
            radiation_losses=losses,
        )
