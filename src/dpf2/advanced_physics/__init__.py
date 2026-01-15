"""Advanced physics modules for DPF2.

This package provides advanced physics capabilities beyond the base
Hall-MHD solver, including:

- **hall_mhd**: 3D Hall-MHD solver with constrained transport
- **kinetic**: Hybrid PIC solver and beam-target fusion models
- **radiation**: Multigroup radiation transport
- **atomic**: Non-LTE ionization models

These modules are designed for research-grade simulations requiring
accurate treatment of multi-physics effects in dense plasma focus
devices.
"""

from __future__ import annotations

from .hall_mhd import HallMHDSolver3D, CTUpdate, whistler_frequency
from .kinetic import HybridPICSolver, Particle, beam_target_yield
from .radiation import MultigroupRadiationSolver
from .atomic import NLTEIonization

__all__ = [
    "HallMHDSolver3D",
    "CTUpdate",
    "whistler_frequency",
    "HybridPICSolver",
    "Particle",
    "beam_target_yield",
    "MultigroupRadiationSolver",
    "NLTEIonization",
]
