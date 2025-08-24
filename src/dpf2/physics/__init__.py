from .mhd import ResistiveMHD

try:  # optional imports may require heavy dependencies (e.g., pydantic)
    from .simple_plasma import ZeroDPlasma
    from .hall_mhd import HallMHD
    from .hooks import neutral_density_source, wall_ablation_source

    __all__ = [
        "ResistiveMHD",
        "ZeroDPlasma",
        "HallMHD",
        "neutral_density_source",
        "wall_ablation_source",
    ]
except Exception:  # pragma: no cover - fallback for minimal environments
    ZeroDPlasma = HallMHD = None  # type: ignore
    neutral_density_source = wall_ablation_source = None  # type: ignore
    __all__ = ["ResistiveMHD"]
