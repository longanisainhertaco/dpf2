from __future__ import annotations

import json
from dataclasses import dataclass

from pathlib import Path
from typing import Dict, Sequence, Callable

from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:  # pragma: no cover - mpi4py may not be available
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - gracefully handle missing MPI
    MPI = None

try:  # pragma: no cover - GPU backend optional
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore

try:  # pragma: no cover - numba optional
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrapper(func):
            return func

        return wrapper

from .dpf_config import DPFConfig
from .circuit_config import CircuitConfig

from .circuit_solver import RLCCircuit, CircuitSolver, run_circuit_simulation
from .core.bases import PlasmaSolverBase, CouplingState, DiagnosticsBase
from .pinch_models import (
    AnalyticPinchModel,
    SemiAnalyticPinchModel,
    PinchModelBase,
    MHDPinchModel,
)

from .physics.energy import EnergyTracker
from .physics.radiation import RadiationTransport, MultiGroupDiffusion


__all__ = ["SimulationEngine", "SimulationResults", "EnsembleResults"]


@njit(cache=True)
def _capacitor_energy(voltage: float, capacitance: float) -> float:
    """Return stored energy in a capacitor using Numba for speed."""
    return 0.5 * capacitance * voltage * voltage


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
    """Execute a minimal Dense Plasma Focus simulation.

    Parameters
    ----------
    config:
        Simulation configuration.
    comm:
        Optional MPI communicator used for synchronising state across
        ranks. If not provided, ``MPI.COMM_WORLD`` is used when
        ``mpi4py`` is available.
    num_threads:
        Number of worker threads used for circuit stepping. A value of
        ``1`` disables multithreading.  These hooks allow heavier
        simulations to leverage multicore machines without altering the
        core algorithm.
    use_gpu:
        When ``True`` and :mod:`cupy` is available, state arrays are
        allocated on the GPU.  Computations fall back to NumPy when the
        library is missing or a GPU is not present.
    """

    def __init__(
        self,
        config: DPFConfig,
        comm: "MPI.Comm | None" = None,
        num_threads: int = 1,
        *,
        use_gpu: bool = False,
    ) -> None:
        self.config = config.resolve_defaults()
        # MPI communicator -------------------------------------------------
        self.comm = comm if comm is not None else (MPI.COMM_WORLD if MPI else None)
        self.rank = self.comm.Get_rank() if self.comm is not None else 0
        # GPU backend ------------------------------------------------------
        self.use_gpu = bool(use_gpu and cp is not None)
        self.xp = cp if self.use_gpu else np
        # Thread pool ------------------------------------------------------
        self._executor: ThreadPoolExecutor | None = None
        if num_threads and num_threads > 1:
            self._executor = ThreadPoolExecutor(max_workers=num_threads)

        # Adaptive mesh refinement driver
        self._mesh = self._setup_mesh()

    # ------------------------------------------------------------------
    def _to_numpy(self, arr: np.ndarray | "cp.ndarray") -> np.ndarray:
        """Convert ``arr`` to a NumPy array regardless of backend."""
        if self.use_gpu and cp is not None:
            return cp.asnumpy(arr)  # type: ignore[no-untyped-call]
        return np.asarray(arr)

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
    def _setup_mesh(self):
        """Instantiate an AMR mesh wrapper when refinement criteria provided."""
        crit = self.config.parallel_settings.amr_refinement_criteria
        if not crit:
            return None
        from .mesh import AMRMesh

        gr = self.config.grid_resolution
        shape = (getattr(gr, "nx", 1), getattr(gr, "ny", 1), getattr(gr, "nz", 1))
        return AMRMesh(shape, crit)

    # ------------------------------------------------------------------
    def run(
        self,
        method: str = "analytical",
        pinch_model: str = "analytic",

        plasma_solver: PlasmaSolverBase | None = None,
        *,
        energy_csv: str | None = None,
        energy_tol: float | None = None,
        diagnostics: Sequence[DiagnosticsBase] | None = None,
        progress_cb: Callable[[int, float, float, float], None] | None = None,
    ) -> SimulationResults:
        """Run the simulation and return aggregated results.

        Parameters
        ----------
        progress_cb:
            Optional callback invoked after each circuit step with
            ``(step, time, current, voltage)`` allowing in-situ diagnostics
            or live visualisation.
        """
        sc = self.config.simulation_control
        dt = sc.min_dt or 1e-9
        t_end = sc.time_end - sc.time_start

        circuit = self._setup_circuit()
        tracker = EnergyTracker()
        radiation = RadiationTransport(MultiGroupDiffusion([0.0]), dx=1.0)

        # Record initial energies
        tracker.add(
            capacitor=_capacitor_energy(circuit.voltages[-1], circuit.circuit.C),
            inductive=0.5 * circuit.circuit.L * circuit.currents[-1] ** 2,
        )

        current = circuit.currents[-1]
        voltage = circuit.voltages[-1]
        step = 0
        plasma_state = None
        while circuit.time[-1] < t_end:
            state = CouplingState(current=current, voltage=voltage)
            if plasma_solver is not None:
                plasma_state = plasma_solver.step(plasma_state, dt, current, voltage)
                iface = plasma_solver.coupling_interface()
                br = iface.back_reaction
                if self.comm is not None and self.comm.size > 1:  # pragma: no cover - MPI
                    br = self.comm.allreduce(br, op=MPI.SUM)
                state.back_reaction = br

            # Optional multithreading for circuit stepping
            if self._executor is not None:
                future = self._executor.submit(
                    circuit.step, state, 0.0, dt, energy_tracker=tracker
                )
                updated = future.result()
            else:
                updated = circuit.step(state, 0.0, dt, energy_tracker=tracker)

            if self.comm is not None and (self.comm.size > 1):  # pragma: no cover - MPI
                updated = self.comm.bcast(updated, root=0)

            # Radiation transport coupling
            rad_in = tracker.thermal[-1] if tracker.thermal else 0.0
            rad_out, rad_groups = radiation.step(rad_in, dt)
            tracker.thermal[-1] = float(rad_out)
            tracker.radiative[-1] = float(np.sum(rad_groups))

            if diagnostics:
                for diag in diagnostics:
                    diag.record(updated, circuit.time[-1])

            if self._mesh is not None:
                self._mesh.refine()

            current, voltage = updated.current, updated.voltage
            step += 1
            if progress_cb is not None:
                progress_cb(step, circuit.time[-1], current, voltage)

        t_xp = self.xp.array(circuit.time)
        current_xp = self.xp.array(circuit.currents)
        t = self._to_numpy(t_xp)
        current_arr = self._to_numpy(current_xp)

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

