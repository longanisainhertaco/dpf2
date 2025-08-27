from .mhd import ResistiveMHD
from .pic_driver import PicDriver

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

__all__ = ["ResistiveMHD"]

if ZeroDPlasma is not None:
    __all__.append("ZeroDPlasma")
if HallMHD is not None:
    __all__.append("HallMHD")
if neutral_density_source is not None:
    __all__.extend(["neutral_density_source", "wall_ablation_source"])
__all__.append("PicDriver")
