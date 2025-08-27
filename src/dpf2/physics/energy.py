from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class EnergyTracker:
    """Accumulate energy components over time."""

    capacitor: list[float] = field(default_factory=list)
    inductive: list[float] = field(default_factory=list)
    kinetic: list[float] = field(default_factory=list)
    thermal: list[float] = field(default_factory=list)
    magnetic: list[float] = field(default_factory=list)
    radiative: list[float] = field(default_factory=list)

    def add(
        self,
        *,
        capacitor: float = 0.0,
        inductive: float = 0.0,
        kinetic: float = 0.0,
        thermal: float = 0.0,
        magnetic: float = 0.0,
        radiative: float = 0.0,
    ) -> None:
        """Record energies for the current time step."""

        self.capacitor.append(float(capacitor))
        self.inductive.append(float(inductive))
        self.kinetic.append(float(kinetic))
        self.thermal.append(float(thermal))
        self.magnetic.append(float(magnetic))
        self.radiative.append(float(radiative))

    @property
    def total(self) -> np.ndarray:
        """Total energy at each recorded step including radiative losses."""

        return (
            np.array(self.capacitor)
            + np.array(self.inductive)
            + np.array(self.kinetic)
            + np.array(self.thermal)
            + np.array(self.magnetic)
            + np.array(self.radiative)
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return all tracked energies as numpy arrays."""

        return {
            "capacitor": np.array(self.capacitor),
            "inductive": np.array(self.inductive),
            "kinetic": np.array(self.kinetic),
            "thermal": np.array(self.thermal),
            "magnetic": np.array(self.magnetic),
            "radiative": np.array(self.radiative),
            "total": self.total,
        }


__all__ = ["EnergyTracker"]
