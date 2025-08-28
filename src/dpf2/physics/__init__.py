from .mhd import ResistiveMHD

from .pic_driver import PicDriver
from .warpx_picmi import WarpXPicmiDriver


# Import optional modules individually so that a failure in one does not
# prevent access to the others.  This keeps ``HallMHD`` available even when
# ``pydantic`` or other heavy dependencies required by unrelated modules are
# missing.

try:  # pragma: no cover - exercised when dependency is available
    from .hall_mhd import HallMHD  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    HallMHD = None  # type: ignore

try:  # pragma: no cover - exercised when dependency is available
    from .simple_plasma import ZeroDPlasma  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    ZeroDPlasma = None  # type: ignore

try:  # pragma: no cover - exercised when dependency is available
    from .hooks import neutral_density_source, wall_ablation_source  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    neutral_density_source = wall_ablation_source = None  # type: ignore

try:  # pragma: no cover - exercised when dependency is available
    from .radiation_mhd_solver import RadiationMHDSolver  # type: ignore
except Exception:  # pragma: no cover - fallback when optional deps missing
    RadiationMHDSolver = None  # type: ignore

__all__ = ["ResistiveMHD", "EnergyTracker"]

if ZeroDPlasma is not None:
    __all__.append("ZeroDPlasma")
if HallMHD is not None:
    __all__.append("HallMHD")
if neutral_density_source is not None:
    __all__.extend(["neutral_density_source", "wall_ablation_source"])
__all__.append("PicDriver")
__all__.append("WarpXPicmiDriver")
if RadiationMHDSolver is not None:
    __all__.append("RadiationMHDSolver")
