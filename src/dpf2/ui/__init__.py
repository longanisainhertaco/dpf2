"""User interface panels."""

from .verification_panel import VerificationPanelUI
from .lab_mode import LabModeUI
from .performance_dashboard import ComparisonDashboard, aggregate_comparisons


__all__ = [
    "VerificationPanelUI",
    "LabModeUI",
    "ComparisonDashboard",
    "aggregate_comparisons",
]

