"""GUI utilities for DPF2."""

from .project_manager import ProjectManager

try:  # pragma: no cover - optional flask dependency
    from .dashboard import launch
    __all__ = ["ProjectManager", "launch"]
except Exception:  # pragma: no cover - simplify when Flask missing
    __all__ = ["ProjectManager"]
