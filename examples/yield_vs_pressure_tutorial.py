"""Yield vs. Pressure optimization tutorial.

This script demonstrates how to perform a simple pressure sweep and overlay the
resulting yields using :class:`dpf2.gui.ProjectManager`.

The example intentionally uses a small parameter grid so that it can run quickly
on modest hardware.  Each sweep performs two simulations with different anode
radii at two pressures and then overlays the yield-versus-pressure trend.
"""

from dpf2.core.config import DPFConfig
from dpf2.gui import ProjectManager


def main() -> None:
    cfg = DPFConfig()
    manager = ProjectManager()

    pressures = [0.5, 1.0]  # torr
    for p in pressures:
        cfg.initial_pressure = p
        manager.run_sweep(f"pressure_{p}", cfg, "anode_radius", [0.01, 0.02])

    manager.overlay_yield_pressure("yield_pressure.png")
    manager.export_metrics("metrics.csv")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
