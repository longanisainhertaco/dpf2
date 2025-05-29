"""Core simulation driver."""
from __future__ import annotations

from pathlib import Path

from ..mesh import Mesh2D
from .config import DPFConfig
from ..io.data_writer import DataWriter


class DPFSimulation:
    """Main class orchestrating a DPF simulation."""

    def __init__(self, config: DPFConfig) -> None:
        self.config = config
        self.mesh = self._setup_mesh()
        self.writer = DataWriter("output")

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

    def run(self, end_time: float | None = None, output_dir: str | None = None) -> None:
        """Placeholder main loop."""
        end = end_time or self.config.end_time
        out = output_dir or "output"
        Path(out).mkdir(parents=True, exist_ok=True)
        # Placeholder for future time loop
        self.writer.write_hdf5({}, time=0.0)
