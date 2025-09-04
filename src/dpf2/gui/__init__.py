"""GUI utilities for DPF2."""

import logging

from .project_manager import ProjectManager

logger = logging.getLogger(__name__)

__all__ = ["ProjectManager"]

try:  # pragma: no cover - optional flask dependency
    from .dashboard import launch
    __all__.append("launch")
except Exception:  # pragma: no cover - simplify when Flask missing
    logger.warning(
        "Flask is required for the dashboard; install with `pip install flask`.",
        exc_info=True,
    )

try:  # pragma: no cover - optional dash dependency
    from .interactive import launch as launch_dash
    __all__.append("launch_dash")
except Exception:  # pragma: no cover - simplify when dash missing
    logger.warning(
        "Dash is required for the interactive GUI; install with `pip install dash`.",
        exc_info=True,
    )

try:  # pragma: no cover - optional PyQt dependency
    from .qt_sweep import launch as launch_qt
    __all__.append("launch_qt")
except Exception:  # pragma: no cover - simplify when PyQt missing
    logger.warning(
        "PyQt5 is required for the Qt sweep GUI; install with `pip install PyQt5`.",
        exc_info=True,
    )
