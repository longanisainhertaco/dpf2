"""Simple GUI panel for running parameter sweeps.

The panel acts as a very lightweight wrapper around the :class:`ProjectManager`
so that engineers can trigger sweeps from a graphical environment without
interfacing directly with the optimization APIs.  Only a minimal subset of the
full project manager functionality is exposed as the aim of this module is to
provide an easily scriptable facade for external tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict

from ..core.config import DPFConfig
from ..gui.project_manager import ProjectManager


@dataclass
class SweepPanelUI:
    """Run parameter sweeps and capture resulting metrics."""

    output_dir: Path = Path("sweep_output")
    project: str = "default"

    def run_sweep(
        self,
        label: str,
        config: DPFConfig,
        parameter: str,
        values: Iterable[float],
    ) -> Dict[float, Dict[str, float]]:
        pm = ProjectManager(project=self.project)
        return pm.run_sweep(label, config, parameter, list(values), output_dir=self.output_dir)


__all__ = ["SweepPanelUI"]
