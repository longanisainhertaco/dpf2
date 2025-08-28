from __future__ import annotations

import json
from dataclasses import dataclass

from pathlib import Path
from typing import Dict


import numpy as np

from .dpf_config import DPFConfig
from .circuit_config import CircuitConfig

from .circuit_solver import RLCCircuit, CircuitSolver, run_circuit_simulation
from .core.bases import PlasmaSolverBase, CouplingState
from .pinch_models import (
    AnalyticPinchModel,
    SemiAnalyticPinchModel,
    PinchModelBase,
    MHDPinchModel,
)

from .physics.energy import EnergyTracker


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
        tracker = EnergyTracker()

        # Record initial energies
        tracker.add(
            capacitor=0.5 * circuit.circuit.C * circuit.voltages[-1] ** 2,
            inductive=0.5 * circuit.circuit.L * circuit.currents[-1] ** 2,
        )

        current = circuit.currents[-1]
        voltage = circuit.voltages[-1]
        while circuit.time[-1] < t_end:
            state = CouplingState(current=current, voltage=voltage)
            updated = circuit.step(state, 0.0, dt, energy_tracker=tracker)
            current, voltage = updated.current, updated.voltage

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

