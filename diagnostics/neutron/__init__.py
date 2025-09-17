"""Neutron diagnostic yield utilities."""

from .thermonuclear import compute_yield as thermonuclear_yield
from .beam_target import compute_yield as beam_target_yield

__all__ = [
    "thermonuclear_yield",
    "beam_target_yield",
]
