"""Core driver for minimal DPF simulations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

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
from .physics.energy import EnergyTracker

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
    energies: Dict[str, np.ndarray] | None = None

    def to_dict(self) -> Dict[str, object]:
        data = {
            "time": self.time.tolist(),
            "current": self.current.tolist(),
            "pinch_radius": self.radius.tolist(),
            "temperature": self.temperature.tolist(),
            "pressure": self.pressure.tolist(),
            "neutron_yield": self.neutron_yield,
            **({"axial_position": self.axial_position.tolist()} if self.axial_position is not None else {}),
        }
        if self.energies is not None:
            data["energies"] = {k: v.tolist() for k, v in self.energies.items()}
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
        energy_csv: str | Path | None = None,
        energy_tol: float | None = None,
    ) -> SimulationResults:
        sc = self.config.simulation_control
        dt = sc.min_dt or 1e-9
        t_end = sc.time_end - sc.time_start
        circuit = self._setup_circuit()
        tracker = EnergyTracker()

        # Record initial energies
        tracker.add(
            capacitor=0.5 * circuit.circuit.C * circuit.voltages[-1] ** 2,
            inductive=0.5 * circuit.circuit.L * circuit.currents[-1] ** 2,
        )

        current = circuit.currents[-1]
        while circuit.time[-1] < t_end:
            current, _ = circuit.step(
                current,
                0.0,
                dt,
                energy_tracker=tracker,
            )

        t = np.array(circuit.time)
        current_arr = np.array(circuit.currents)

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
        pres = plasma.run(t, current_arr)

        energies = tracker.as_dict()

        if energy_csv is not None:
            energy_path = Path(energy_csv)
            import csv

            with energy_path.open("w", newline="") as f:
                writer = csv.writer(f)
                keys = ["capacitor", "inductive", "kinetic", "thermal", "magnetic", "radiative", "total"]
                writer.writerow(["time"] + keys)
                for i, ti in enumerate(t):
                    writer.writerow([ti] + [energies[k][i] for k in keys])

        # Emit summary table
        keys = ["capacitor", "inductive", "kinetic", "thermal", "magnetic", "radiative", "total"]
        print("Energy summary (J):")
        for k in keys:
            arr = energies[k]
            print(f"{k:>10}: {arr[0]:12.3e} -> {arr[-1]:12.3e}")

        total = energies["total"]
        if energy_tol is not None:
            if total[0] != 0.0:
                rel_err = abs(total[-1] - total[0]) / abs(total[0])
            else:
                rel_err = 0.0
            if rel_err > energy_tol:
                raise AssertionError(
                    f"Energy non-conservation {rel_err:.2%} exceeds tolerance {energy_tol:.2%}"
                )

        return SimulationResults(
            time=pres.time,
            current=current_arr,
            radius=pres.radius,
            temperature=pres.temperature,
            pressure=pres.pressure,
            neutron_yield=pres.neutron_yield,
            axial_position=pres.axial_position,
            energies=energies,
        )
