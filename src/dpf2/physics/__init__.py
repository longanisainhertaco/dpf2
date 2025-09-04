from .mhd import ResistiveMHD
from .energy import EnergyTracker
from .pic import SimplePIC, HybridPIC
from .eos import TabulatedEOS, load_tabulated_eos, load_standard_eos
from .radiation import RadiationTransport
from .pic_driver import PicDriver, SimplePicDriver, WarpXPICDriver
from .lower_hybrid_drift import LowerHybridDrift
from .m0_instability import MZeroInstability


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


__all__ = [
    "ResistiveMHD",
    "EnergyTracker",
    "SimplePIC",
    "HybridPIC",
    "TabulatedEOS",
    "load_tabulated_eos",
    "load_standard_eos",
    "RadiationTransport",
    "LowerHybridDrift",
    "MZeroInstability",
]


if ZeroDPlasma is not None:
    __all__.append("ZeroDPlasma")
if HallMHD is not None:
    __all__.append("HallMHD")
if neutral_density_source is not None:
    __all__.extend(["neutral_density_source", "wall_ablation_source"])
__all__.append("PicDriver")
__all__.append("SimplePicDriver")
__all__.append("WarpXPICDriver")
