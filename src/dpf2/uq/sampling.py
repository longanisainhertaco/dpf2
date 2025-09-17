"""Deprecated module. Use :mod:`dpf2.uq.samplers` instead."""

from __future__ import annotations

from .samplers import latin_hypercube, sobol_sample

__all__ = ["latin_hypercube", "sobol_sample"]
