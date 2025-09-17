import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import warnings

try:  # optional SciPy dependency
    from scipy.constants import epsilon_0, e
except Exception:  # pragma: no cover - fallback values
    epsilon_0 = 8.854187817e-12
    e = 1.602176634e-19


def compute_debye_length(temperature_eV: float, density_m3: float) -> float:
    """Return the electron Debye length in metres.

    Parameters
    ----------
    temperature_eV:
        Electron temperature in electronvolts.
    density_m3:
        Electron number density in m^-3.
    """
    if temperature_eV <= 0:
        raise ValueError("temperature_eV must be positive")
    if density_m3 <= 0:
        raise ValueError("density_m3 must be positive")
    return math.sqrt(epsilon_0 * temperature_eV / (density_m3 * e))


def plasma_inductance_circuit(
    voltage: float, current: float, resistance: float, dI_dt: float
) -> float:
    r"""Compute effective inductance from circuit quantities.

    Implements :math:`L = (V - I R) / \dot{I}`.  ``dI_dt`` must be non-zero.
    """
    if dI_dt == 0:
        raise ValueError("dI/dt must be non-zero")
    return (voltage - current * resistance) / dI_dt


def check_thresholds(
    dt: float,
    debye_length: float,
    cell_size: float,
    particles_per_cell: int,
    *,
    max_dt: float,
    min_debye_cells: float,
    min_particles_per_cell: int,
) -> List[str]:
    """Check basic numerical stability thresholds.

    Returns a list of human-readable warning messages.  Each message is also
    emitted via :mod:`warnings` so they appear in CLI output.
    """
    warnings_list: List[str] = []
    if dt > max_dt:
        warnings_list.append(
            f"timestep {dt:.3e}s exceeds recommended maximum {max_dt:.3e}s"
        )
    if debye_length < min_debye_cells * cell_size:
        warnings_list.append(
            f"Debye length {debye_length:.3e}m below {min_debye_cells:.2f} cell widths ({cell_size:.3e}m)"
        )
    if particles_per_cell < min_particles_per_cell:
        warnings_list.append(
            f"particles per cell {particles_per_cell} below recommended minimum {min_particles_per_cell}"
        )
    for msg in warnings_list:
        warnings.warn(msg)
    return warnings_list


__all__ = [
    "compute_debye_length",
    "plasma_inductance_circuit",
    "check_thresholds",
    "ThresholdDashboard",
]


@dataclass
class ThresholdDashboard:
    """Record threshold metrics and feed them to a JSON dashboard.

    The dashboard stores per-step metrics and their colour-coded status for
    easy consumption by a GUI or CLI monitor.  Thresholds are user-configurable
    and violations can optionally abort the run.
    """

    output_dir: Path = Path("synthetic_diagnostics/thresholds")
    max_cfl: float | None = None
    min_lambda_D_dx: float | None = None
    max_divB: float | None = None
    abort_on_violation: bool = False
    history: list[dict[str, float | str]] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def _status(
        self, value: float, threshold: float | None, *, higher_is_better: bool
    ) -> str:
        if threshold is None:
            return "grey"
        if higher_is_better:
            if value < threshold:
                return "red"
            if value < 1.2 * threshold:
                return "yellow"
            return "green"
        if value > threshold:
            return "red"
        if value > 0.8 * threshold:
            return "yellow"
        return "green"

    def log(
        self,
        *,
        step: int,
        cfl: float,
        lambda_D: float,
        cell_size: float,
        divB: float,
    ) -> dict[str, str]:
        """Record metrics and warn/abort on threshold violations."""

        lambda_ratio = lambda_D / cell_size
        statuses = {
            "cfl": self._status(cfl, self.max_cfl, higher_is_better=False),
            "lambda_D_dx": self._status(
                lambda_ratio, self.min_lambda_D_dx, higher_is_better=True
            ),
            "divB": self._status(divB, self.max_divB, higher_is_better=False),
        }

        entry = {
            "step": step,
            "cfl": cfl,
            "cfl_status": statuses["cfl"],
            "lambda_D_dx": lambda_ratio,
            "lambda_D_dx_status": statuses["lambda_D_dx"],
            "divB": divB,
            "divB_status": statuses["divB"],
        }
        self.history.append(entry)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "dashboard.json", "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)

        if statuses["cfl"] == "red" and self.max_cfl is not None:
            self.logger.warning(f"CFL above threshold: {cfl:g} > {self.max_cfl:g}")
        if statuses["lambda_D_dx"] == "red" and self.min_lambda_D_dx is not None:
            self.logger.warning(
                f"lambda_D/dx below threshold: {lambda_ratio:g} < {self.min_lambda_D_dx:g}"
            )
        if statuses["divB"] == "red" and self.max_divB is not None:
            self.logger.warning(f"divB above threshold: {divB:g} > {self.max_divB:g}")
        if self.abort_on_violation and any(s == "red" for s in statuses.values()):
            raise RuntimeError("Threshold violation")
        return statuses
