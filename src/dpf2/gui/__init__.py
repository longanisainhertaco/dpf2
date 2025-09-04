"""GUI utilities for DPF2."""

from .project_manager import ProjectManager

__all__ = ["ProjectManager"]

try:  # pragma: no cover - optional flask dependency
    from .dashboard import launch
    __all__.append("launch")
except Exception:  # pragma: no cover - simplify when Flask missing
    pass

try:  # pragma: no cover - optional dash dependency
    from .interactive import launch as launch_dash
    __all__.append("launch_dash")
except Exception:  # pragma: no cover - simplify when dash missing
    pass
