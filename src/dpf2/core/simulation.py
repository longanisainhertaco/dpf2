"""Core simulation driver."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict

from ..mesh import Mesh2D
from .config import DPFConfig
from ..io.data_writer import DataWriter

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Container bundling simulation traces and derived metrics."""

    times: list[float]
    currents: list[float]
    voltages: list[float]
    metrics: Dict[str, float]


class DPFSimulation:
    """Main class orchestrating a DPF simulation."""

    def __init__(
        self,
        config: DPFConfig,
        plasma_solver: Any | None = None,
        circuit_solver: Any | None = None,
    ) -> None:
        self.config = config
        self.mesh = self._setup_mesh()
        self.plasma_solver = plasma_solver
        self.circuit_solver = circuit_solver
        self.writer: DataWriter | None = None

        # Runtime state variables
        self.time = 0.0
        self.plasma_state: Any = 0.0
        self.current = 0.0
        self.voltage = self.config.charging_voltage
        self.run_outputs: list[dict[str, float]] = []

    def _setup_mesh(self) -> Mesh2D:
        cfg = self.config
        return Mesh2D(
            0.0,
            cfg.anode_radius,
            0.0,
            cfg.electrode_length,
            cfg.nr_cells,
            cfg.nz_cells,
        )

    def run(
        self,
        end_time: float | None = None,
        output_dir: str | None = None,
        output_interval: float | None = None,
        seeds: Dict[str, int] | None = None,
        verbose: bool = False,
        progress_cb: Callable[[int, float], None] | None = None,
        *,
        return_metrics: bool = False,
    ) -> tuple[list[float], list[float], list[float]] | SimulationResult:
        """Advance the simulation until ``end_time``.

        Parameters
        ----------
        end_time:
            Optional final time.  Defaults to ``self.config.end_time``.
        output_dir:
            Directory where output files are written.
        output_interval:
            Time between data dumps.  Defaults to ``end_time`` (only final
            state).
        verbose:
            If ``True`` prints progress and simple energy diagnostics.

        Returns
        -------
        (times, currents, voltages): tuple[list[float], list[float], list[float]]
            Time history of the main circuit quantities for quick-look plotting
            or diagnostics. When ``return_metrics`` is ``True`` a
            :class:`SimulationResult` bundle including derived metrics is
            returned instead.
        """

        end = end_time or self.config.end_time
        interval = output_interval or end
        t_start = time.perf_counter()

        out = output_dir
        last_output = self.time
        initial_outputs = self._collect_outputs()
        if out is not None:
            Path(out).mkdir(parents=True, exist_ok=True)
            self.writer = DataWriter(out, config=asdict(self.config), seeds=seeds)
            self.writer.write_hdf5(initial_outputs, time=self.time)
        else:
            self.writer = None
        self.run_outputs.append({"time": self.time, **initial_outputs})

        times = [self.time]
        currents = [self.current]
        voltages = [self.voltage]

        step = 0
        while self.time < end:
            dt = min(
                self.config.cfl_number * min(self.mesh.dr, self.mesh.dz),
                end - self.time,
            )

            feedback = None
            if self.plasma_solver is not None:
                result = self.plasma_solver.step(
                    self.plasma_state, dt, self.current, self.voltage
                )
                if isinstance(result, tuple):
                    self.plasma_state, feedback = result
                else:
                    self.plasma_state = result
                    feedback = getattr(self.plasma_solver, "circuit_feedback", None)
            if self.circuit_solver is not None:
                self.current, self.voltage = self.circuit_solver.step(
                    self.current, self.voltage, dt, feedback
                )

            self.time += dt
            step += 1

            times.append(self.time)
            currents.append(self.current)
            voltages.append(self.voltage)

            if verbose:
                energy = 0.5 * self.config.capacitance * self.voltage**2 + 0.5 * self.config.inductance * self.current**2
                logger.info(
                    "t=%g s I=%g A V=%g V energy=%g J", self.time, self.current, self.voltage, energy
                )

            if progress_cb is not None:
                progress_cb(step, self.time)

            if (self.time - last_output) >= interval or self.time >= end:
                outputs = self._collect_outputs(feedback)
                if self.writer is not None:
                    self.writer.write_hdf5(outputs, time=self.time)
                self.run_outputs.append({"time": self.time, **outputs})
                last_output = self.time

        return times, currents, voltages

    def _collect_outputs(self, feedback: Any | None = None) -> Dict[str, float]:
        """Assemble a snapshot of run outputs for diagnostics."""

        outputs: Dict[str, float] = {
            "current": float(self.current),
            "voltage": float(self.voltage),
        }
        if feedback is not None:
            outputs["plasma_inductance"] = float(getattr(feedback, "Lp", 0.0))
        if self.plasma_solver is not None and hasattr(self.plasma_solver, "effective_impedance"):
            try:
                outputs["effective_impedance"] = float(self.plasma_solver.effective_impedance())  # type: ignore[call-arg]
            except Exception:
                pass
        return outputs
        if not return_metrics:
            return times, currents, voltages

        energy = 0.5 * self.config.capacitance * self.voltage**2 + 0.5 * self.config.inductance * self.current**2
        runtime = float(time.perf_counter() - t_start)
        metrics = {
            "peak_current": max(currents) if currents else 0.0,
            "pinch_time": times[currents.index(max(currents))] if currents else 0.0,
            "yield": (max(currents) ** 2) / (self.config.anode_radius * self.config.initial_pressure)
            if self.config.anode_radius > 0 and self.config.initial_pressure > 0 and currents
            else 0.0,
            "runtime_s": runtime,
            "wall_plug_efficiency": ((max(currents) ** 2) / (self.config.anode_radius * self.config.initial_pressure))
            / energy
            if energy > 0 and self.config.anode_radius > 0 and self.config.initial_pressure > 0 and currents
            else 0.0,
            "yield_per_hour": ((max(currents) ** 2) / (self.config.anode_radius * self.config.initial_pressure))
            / runtime
            * 3600.0
            if runtime > 0 and self.config.anode_radius > 0 and self.config.initial_pressure > 0 and currents
            else 0.0,
            "S": max(currents) / (self.config.anode_radius * self.config.initial_pressure)
            if self.config.anode_radius > 0 and self.config.initial_pressure > 0 and currents
            else 0.0,
        }

        return SimulationResult(times=times, currents=currents, voltages=voltages, metrics=metrics)
