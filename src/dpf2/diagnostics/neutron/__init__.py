"""High level neutron diagnostic utilities.

This subpackage provides convenience wrappers around the existing neutron
analysis helpers in :mod:`dpf2.diagnostics.neutron_spectra` and
:mod:`dpf2.diagnostics.neutron_yield`.  The functions exposed here focus on
common analysis tasks such as computing beam–target and thermonuclear yields,
aggregating angular distributions and generating synthetic time-of-flight (ToF)
signals correlated with circuit ``I``–``V`` traces.
"""

from .base import (
    Detector,
    DetectorLayout,
    synthetic_tof_spectrum,
    angular_spectrum,
    anisotropy_metric,
    forward_radial_backward_counts,
    anisotropy_ratios,
    load_detector_layout,
    load_response,
    apply_response,
    anisotropy_report,
)
from .yield_calculators import thermonuclear_yield, beam_target_yield
from .angular import angular_distribution
from .tof import synthetic_tof_correlated
from .benchmarks import compare_with_benchmark

__all__ = [
    "Detector",
    "DetectorLayout",
    "synthetic_tof_spectrum",
    "angular_spectrum",
    "anisotropy_metric",
    "forward_radial_backward_counts",
    "anisotropy_ratios",
    "load_detector_layout",
    "load_response",
    "apply_response",
    "anisotropy_report",
    "thermonuclear_yield",
    "beam_target_yield",
    "angular_distribution",
    "synthetic_tof_correlated",
    "compare_with_benchmark",
]
