from __future__ import annotations

"""Modular radiation transport helpers."""

from dataclasses import dataclass
from typing import Sequence, Tuple

# ``MultiGroupDiffusion`` is optional; provide a minimal stub for test
# environments where the radiation package is stripped down.
try:  # pragma: no cover - exercised when package is present
    from ..radiation.multigroup import MultiGroupDiffusion  # type: ignore
except Exception:  # pragma: no cover - fallback for stripped environments
    class MultiGroupDiffusion:  # type: ignore
        def __init__(self, opacities):
            self.opacities = list(opacities)
            self.group_count = len(self.opacities)
            self.energy = [[0.0] for _ in range(self.group_count)]

        def diffuse(self, dx, dt):
            return self.energy

        def couple(self, fluid_energy, dt):
            return fluid_energy


@dataclass
class RadiationTransport:
    """High-level driver for multi-group radiation diffusion."""

    diffusion: MultiGroupDiffusion
    dx: float

    def step(
        self, fluid_energy: Sequence[float] | float, dt: float
    ) -> Tuple[Sequence[float] | float, Sequence[Sequence[float]]]:
        """Diffuse radiation and exchange energy with ``fluid_energy``."""

        updated = self.diffusion.couple(fluid_energy, dt)
        self.diffusion.diffuse(self.dx, dt)
        return updated, self.diffusion.energy


__all__ = ["RadiationTransport", "MultiGroupDiffusion"]
