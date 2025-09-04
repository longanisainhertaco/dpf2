"""Visualization helpers for DPF2.

Currently this subpackage exposes a small routine for animating
plasma sheath evolution.  The functionality is intentionally light
weight so it can be used in examples and tests without requiring the
full simulation stack.
"""

from .sheath import animate_sheath
from .widgets import sheath_widget

__all__ = ["animate_sheath", "sheath_widget"]

