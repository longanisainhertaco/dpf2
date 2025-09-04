"""Synthetic diagnostics package exposing public API and modes submodule."""

from . import core, modes
from .core import *  # noqa: F401,F403

__all__ = [*core.__all__, "modes"]
