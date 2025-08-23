"""Core driver for minimal DPF simulations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np

from dpf_config import DPFConfig
from circuit_config import CircuitConfig

from neutron_yield_model import (
    BoschHaleTable,
    DEFAULT_BOSCH_HALE_TABLE,
    compute_dd_yield,
)

from .circuit_solver import RLCCircuit, CircuitSolver
from .pinch_models import AnalyticPinchModel, SemiAnalyticPinchModel, PinchModelBase

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

        table = BoschHaleTable.from_csv(DEFAULT_BOSCH_HALE_TABLE)
        if pres.axial_position is not None:
            length = pres.axial_position
        else:
            length = np.full_like(pres.radius, getattr(plasma, "length", 1.0))
        volume = np.pi * pres.radius ** 2 * length
        n_i = np.full_like(pres.radius, getattr(plasma, "ion_density", 1e20))
        neutron_yield = compute_dd_yield(
            pres.time, pres.temperature, n_i, volume, table
        )

        return SimulationResults(
            time=pres.time,
            current=current,
            radius=pres.radius,
            temperature=pres.temperature,
            pressure=pres.pressure,
            neutron_yield=neutron_yield,
            axial_position=pres.axial_position,
        )
