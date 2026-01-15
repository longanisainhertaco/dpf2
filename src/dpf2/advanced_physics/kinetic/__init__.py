"""Kinetic physics models including hybrid PIC and beam-target fusion.

This module provides particle-based kinetic simulations and beam-target
fusion yield calculations for dense plasma focus devices.
"""

from __future__ import annotations

from .hybrid_pic import HybridPICSolver, Particle, ParticleSpecies
from .beam_target_fusion import (
    dd_fusion_cross_section,
    dt_fusion_cross_section,
    beam_target_yield,
    BeamTargetModel,
)

__all__ = [
    "HybridPICSolver",
    "Particle",
    "ParticleSpecies",
    "dd_fusion_cross_section",
    "dt_fusion_cross_section",
    "beam_target_yield",
    "BeamTargetModel",
]
