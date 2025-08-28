"""Core simulation driver."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import logging

from ..mesh import Mesh2D
from .config import DPFConfig
from ..io.data_writer import DataWriter

logger = logging.getLogger(__name__)


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
        verbose: bool = False,
        progress_cb: Callable[[int, float], None] | None = None,
    ) -> tuple[list[float], list[float], list[float]]:
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
            or diagnostics.
        """

        end = end_time or self.config.end_time
        out = output_dir or "output"
        interval = output_interval or end

        Path(out).mkdir(parents=True, exist_ok=True)
        self.writer = DataWriter(out)

        # Write initial state
        self.writer.write_hdf5({"current": self.current, "voltage": self.voltage}, time=self.time)
        last_output = self.time

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
                self.writer.write_hdf5(
                    {"current": self.current, "voltage": self.voltage},
                    time=self.time,
                )
                last_output = self.time

        return times, currents, voltages
