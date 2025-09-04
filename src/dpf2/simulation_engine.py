from __future__ import annotations

import json
from dataclasses import dataclass
import logging

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
from .diagnostics import NeutronYieldStreamer, XRayEmissionStreamer
from .pinch_models import (
    AnalyticPinchModel,
    SemiAnalyticPinchModel,
    PinchModelBase,
    MHDPinchModel,
)

from .physics.energy import EnergyTracker


logger = logging.getLogger(__name__)


__all__ = ["SimulationEngine", "SimulationResults", "EnsembleResults"]


@njit(cache=True)
def _capacitor_energy(voltage: float, capacitance: float) -> float:
    """Return stored energy in a capacitor using Numba for speed."""
    return 0.5 * capacitance * voltage * voltage


@dataclass
class SimulationResults:
    time: np.ndarray
    current: np.ndarray
    voltage: np.ndarray
    radius: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    neutron_yield: float
    axial_position: np.ndarray | None = None
    energies: Dict[str, np.ndarray] | None = None
    dt: float | None = None
    cell_size: float | None = None
    particles_per_cell: float | None = None
    Lp_field: np.ndarray | None = None
    Lp_circuit: np.ndarray | None = None

    def to_dict(self) -> Dict[str, object]:
        data = {
            "time": self.time.tolist(),
            "current": self.current.tolist(),
            "voltage": self.voltage.tolist(),
            "pinch_radius": self.radius.tolist(),
            "temperature": self.temperature.tolist(),
            "pressure": self.pressure.tolist(),
            "neutron_yield": self.neutron_yield,
            **({"axial_position": self.axial_position.tolist()} if self.axial_position is not None else {}),
        }
        if self.energies is not None:
            data["energies"] = {k: v.tolist() for k, v in self.energies.items()}
        if self.dt is not None:
            data["dt"] = self.dt
        if self.cell_size is not None:
            data["cell_size"] = self.cell_size
        if self.particles_per_cell is not None:
            data["particles_per_cell"] = self.particles_per_cell
        if self.Lp_field is not None:
            data["Lp_field"] = self.Lp_field.tolist()
        if self.Lp_circuit is not None:
            data["Lp_circuit"] = self.Lp_circuit.tolist()
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

        # Basic simulation metrics
        self.dt: float | None = None
        self.cell_size: float | None = None
        self.particles_per_cell: float | None = None

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
    def _generate_convergence_plot(self, solver) -> None:
        """Generate a simple convergence plot for PIC runs.

        The plot uses the solver's built-in convergence study to evaluate
        energy conservation at increasingly fine grid resolutions.  Any
        exceptions are silently ignored so that plotting never interrupts
        the main simulation workflow.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - matplotlib optional
            return

        try:
            base_shape = (getattr(solver, "nx", 0), getattr(solver, "ny", 0), getattr(solver, "nz", 0))
            if not all(base_shape):
                return
            resolutions = [base_shape, tuple(2 * s for s in base_shape)]
            energies = solver.run_convergence_study(resolutions)
            if not energies:
                return
            cells = [int(np.prod(r)) for r in resolutions[: len(energies)]]
            plt.figure()
            plt.loglog(cells, energies, marker="o")
            plt.xlabel("Total cells")
            plt.ylabel("Total energy")
            plt.title("PIC convergence study")
            plt.savefig("pic_convergence.png")
            plt.close()
        except (ValueError, RuntimeError):
            logger.exception("Failed to generate PIC convergence plot")

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
        neutron_cb: Callable[[float, float], None] | None = None,
        xray_cb: Callable[[float, float], None] | None = None,
        progress_cb: Callable[[int, float, float, float], None] | None = None,
        hdf5_path: str | None = None,
    ) -> SimulationResults:
        """Run the simulation and return aggregated results.

        Parameters
        ----------
        progress_cb:
            Optional callback invoked after each circuit step with
            ``(step, time, current, voltage)`` allowing in-situ diagnostics
            or live visualisation.
        neutron_cb, xray_cb:
            Callbacks receiving ``(time, value)`` for streaming neutron
            yield and X-ray emission respectively.
        hdf5_path:
            Optional path to an HDF5 file storing ``Lp_field`` and
            ``Lp_circuit`` time series along with run metadata.
        """
        sc = self.config.simulation_control
        dt = sc.min_dt or 1e-9
        t_end = sc.time_end - sc.time_start

        # Basic grid metrics used for threshold checks
        gr = self.config.grid_resolution
        cell_size = min(gr.cell_sizes()) if gr is not None else 0.0
        particles_per_cell: float = 0.0

        # If a PIC solver is supplied, pull resolution and particle counts
        if plasma_solver is not None:
            try:  # Lazy import to avoid heavy dependency unless needed
                from .simulation.pic_solver import PICSolver  # type: ignore
            except Exception:  # pragma: no cover - PICSolver optional
                PICSolver = None  # type: ignore

            if 'PICSolver' in locals() and isinstance(plasma_solver, PICSolver):
                dt = getattr(plasma_solver, 'dt', dt)
                cell_size = min(getattr(plasma_solver, 'dx', cell_size), getattr(plasma_solver, 'dy', cell_size), getattr(plasma_solver, 'dz', cell_size))
                total_cells = getattr(plasma_solver, 'nx', 1) * getattr(plasma_solver, 'ny', 1) * getattr(plasma_solver, 'nz', 1)
                try:
                    total_particles = sum(spec['pos'].shape[0] for spec in getattr(plasma_solver, 'species', {}).values())
                except Exception:
                    total_particles = 0
                if total_cells > 0:
                    particles_per_cell = total_particles / total_cells
                # Compute basic plasma length scales and warn if thresholds violated
                try:
                    from .diagnostics.thresholds import (
                        compute_debye_length,
                        check_thresholds,
                    )
                    ic = self.config.initial_conditions
                    debye = compute_debye_length(ic.temperature, ic.density)
                    max_dt = cell_size / PICSolver.c if cell_size > 0 else dt
                    check_thresholds(
                        dt,
                        debye,
                        cell_size,
                        int(particles_per_cell),
                        max_dt=max_dt,
                        min_debye_cells=1.0,
                        min_particles_per_cell=10,
                    )
                except (ValueError, RuntimeError):
                    logger.exception("Plasma threshold evaluation failed")

                # Attempt to generate a convergence plot for PIC runs
                self._generate_convergence_plot(plasma_solver)

        # Persist metrics for downstream use
        self.dt = dt
        self.cell_size = cell_size
        self.particles_per_cell = particles_per_cell

        circuit = self._setup_circuit()
        tracker = EnergyTracker()

        # Record initial energies
        tracker.add(
            capacitor=_capacitor_energy(circuit.voltages[-1], circuit.circuit.C),
            inductive=0.5 * circuit.circuit.L * circuit.currents[-1] ** 2,
        )

        current = circuit.currents[-1]
        voltage = circuit.voltages[-1]
        step = 0
        plasma_state = None
        coupling = CouplingState(current=current, voltage=voltage)
        diag_list = list(diagnostics or [])
        if neutron_cb is not None:
            diag_list.append(NeutronYieldStreamer(neutron_cb))
        if xray_cb is not None:
            diag_list.append(XRayEmissionStreamer(xray_cb))

        field_lp: list[float] = [0.0]
        circuit_lp: list[float] = [0.0]

        while circuit.time[-1] < t_end:
            if plasma_solver is not None:
                plasma_state = plasma_solver.step(
                    plasma_state, dt, coupling.current, coupling.voltage
                )
                iface = plasma_solver.coupling_interface()
                Lp = getattr(iface, "Lp", 0.0)
                if Lp == 0.0:
                    if hasattr(plasma_solver, "compute_plasma_inductance"):
                        try:
                            Lp = float(
                                plasma_solver.compute_plasma_inductance(
                                    plasma_state, coupling.current
                                )
                            )
                        except Exception:
                            Lp = 0.0
                    elif hasattr(plasma_solver, "plasma_inductance"):
                        try:
                            Lp = float(plasma_solver.plasma_inductance(plasma_state))
                        except Exception:
                            Lp = 0.0
                coupling.Lp = Lp
                coupling.emf = getattr(iface, "emf", 0.0)
                coupling.mutual_inductance = getattr(iface, "mutual_inductance", 0.0)
                br = getattr(iface, "back_reaction", 0.0)
                if self.comm is not None and self.comm.size > 1:  # pragma: no cover - MPI
                    br = self.comm.allreduce(br, op=MPI.SUM)
                coupling.back_reaction = br

            # Optional multithreading for circuit stepping
            if self._executor is not None:
                future = self._executor.submit(
                    circuit.step, coupling, 0.0, dt, energy_tracker=tracker
                )
                updated = future.result()
            else:
                updated = circuit.step(coupling, 0.0, dt, energy_tracker=tracker)

            # Log plasma inductance from solver and inferred from circuit
            field_lp.append(coupling.Lp)
            dIdt = (updated.current - current) / dt if dt != 0 else 0.0
            if dIdt != 0.0:
                num = (
                    circuit.circuit.V0
                    - circuit.circuit.R * current
                    - voltage
                    - coupling.emf
                )
                inferred = num / dIdt - circuit.circuit.L
            else:
                inferred = circuit_lp[-1]
            circuit_lp.append(float(inferred))

            if self.comm is not None and (self.comm.size > 1):  # pragma: no cover - MPI
                updated = self.comm.bcast(updated, root=0)

            coupling.current = updated.current
            coupling.voltage = updated.voltage
            current, voltage = updated.current, updated.voltage

            # Radiation transport coupling (placeholder)
            rad_in = tracker.thermal[-1] if tracker.thermal else 0.0
            tracker.thermal[-1] = float(rad_in)
            tracker.radiative[-1] = 0.0

            if diag_list:
                for diag in diag_list:
                    diag.record(updated, circuit.time[-1])

            if self._mesh is not None:
                self._mesh.refine()

            current, voltage = updated.current, updated.voltage
            step += 1
            if progress_cb is not None:
                progress_cb(step, circuit.time[-1], current, voltage)

        t_xp = self.xp.array(circuit.time)
        current_xp = self.xp.array(circuit.currents)
        voltage_xp = self.xp.array(circuit.voltages)
        t = self._to_numpy(t_xp)
        current_arr = self._to_numpy(current_xp)
        voltage_arr = self._to_numpy(voltage_xp)
        field_lp_arr = np.asarray(field_lp)
        circuit_lp_arr = np.asarray(circuit_lp)

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

        if hdf5_path is not None:
            try:
                import h5py  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("h5py is required for HDF5 output") from exc
            import json
            from .io.data_writer import DataWriter

            meta = {
                "config": self.config.model_dump(),
                "config_hash": DataWriter._hash_config(self.config.model_dump()),
                "git_commit": DataWriter._git_commit(),
            }
            with h5py.File(hdf5_path, "w") as h5:
                h5.create_dataset("time", data=t)
                h5.create_dataset("Lp_field", data=field_lp_arr)
                h5.create_dataset("Lp_circuit", data=circuit_lp_arr)
                h5.create_dataset("metadata", data=json.dumps(meta))

        return SimulationResults(
            time=pres.time,
            current=current_arr,
            voltage=voltage_arr,
            radius=pres.radius,
            temperature=pres.temperature,
            pressure=pres.pressure,
            neutron_yield=pres.neutron_yield,
            axial_position=pres.axial_position,
            energies=energies,
            dt=self.dt,
            cell_size=self.cell_size,
            particles_per_cell=self.particles_per_cell,
            Lp_field=field_lp_arr,
            Lp_circuit=circuit_lp_arr,
        )

    # ------------------------------------------------------------------
    def sweep_ppc_grid(
        self,
        ppc_values: Sequence[int],
        grid_sizes: Sequence[int],
        *,
        method: str = "analytical",
        pinch_model: str = "analytic",
    ) -> Dict[str, object]:
        """Sweep particles-per-cell and grid resolution computing yield variance.

        Parameters
        ----------
        ppc_values:
            Iterable of particles per cell counts.
        grid_sizes:
            Iterable of grid resolutions (``nx = ny = nz``).
        method, pinch_model:
            Forwarded to :meth:`run` for each simulation.

        Returns
        -------
        dict
            Dictionary with ``yields`` and overall ``variance``.
        """

        yields: list[float] = []
        for ppc in ppc_values:
            for n in grid_sizes:
                cfg = self.config.model_copy(deep=True)
                if hasattr(cfg, "physics") and hasattr(cfg.physics, "particles_per_cell"):
                    cfg.physics.particles_per_cell = ppc
                if hasattr(cfg, "grid_resolution"):
                    cfg.grid_resolution.nx = n
                    cfg.grid_resolution.ny = n
                    cfg.grid_resolution.nz = n

                engine = SimulationEngine(
                    cfg,
                    comm=self.comm,
                    num_threads=1,
                    use_gpu=self.use_gpu,
                )
                res = engine.run(method=method, pinch_model=pinch_model)
                yields.append(res.neutron_yield)
                logger.info(
                    "ppc=%s grid=%s yield=%g", ppc, n, res.neutron_yield
                )

        variance = float(np.var(yields)) if yields else 0.0
        out_dir = Path("synthetic_diagnostics/quality")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "ppc_grid_sweep.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "ppc": list(ppc_values),
                    "grid": list(grid_sizes),
                    "yields": yields,
                    "variance": variance,
                },
                fh,
                indent=2,
            )

        logger.info("Yield variance across sweep: %g", variance)
        return {"yields": yields, "variance": variance}

