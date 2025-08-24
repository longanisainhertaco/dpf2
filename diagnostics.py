"""Compatibility wrapper for diagnostics module.

This module re-exports the implementation from :mod:`dpf2.diagnostics`
to maintain backwards compatibility with older import paths that
expected a top-level ``diagnostics`` module.
"""

from dpf2.diagnostics import *  # noqa: F401,F403

