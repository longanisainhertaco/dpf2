import math
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

def plasma_inductance_circuit(voltage: float, current: float, resistance: float, dI_dt: float) -> float:
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
]
