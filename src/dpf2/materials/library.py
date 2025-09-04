from dataclasses import dataclass
from typing import Dict
import math


@dataclass(frozen=True)
class Material:
    """Static material properties."""

    name: str
    density: float  # kg/m^3
    atomic_mass: float  # atomic mass units
    sputter_yield: float  # atoms removed per incident particle
    resistivity: float | None = None  # ohm-m at ``frequency_ref``
    frequency_ref: float = 1.0  # reference frequency for resistivity [Hz]
    surface_conditioning: float | None = None  # multiplier for breakdown field

    def resistivity_at(self, frequency: float) -> float:
        """Return resistivity at ``frequency`` using a skin-effect model."""
        if self.resistivity is None:
            raise ValueError("Material does not have resistivity data")
        if frequency <= 0:
            return self.resistivity
        return self.resistivity * math.sqrt(frequency / self.frequency_ref)

    def conditioned_field(self, base: float) -> float:
        """Apply surface conditioning factor to ``base`` field."""
        if self.surface_conditioning is None:
            return base
        return base * self.surface_conditioning


class MaterialLibrary:
    """Simple registry of materials with basic properties."""

    _materials: Dict[str, Material] = {
        "copper": Material(
            "copper",
            density=8960.0,
            atomic_mass=63.546,
            sputter_yield=0.01,
            resistivity=1.68e-8,
            frequency_ref=1e5,
        ),
        "tungsten": Material(
            "tungsten",
            density=19300.0,
            atomic_mass=183.84,
            sputter_yield=0.005,
            resistivity=5.6e-8,
            frequency_ref=1e5,
        ),
        "stainless_steel": Material(
            "stainless_steel",
            density=8000.0,
            atomic_mass=55.845,
            sputter_yield=0.02,
            resistivity=7.4e-7,
            frequency_ref=1e5,
        ),
        "quartz": Material(
            "quartz",
            density=2648.0,
            atomic_mass=60.08,
            sputter_yield=0.0,
            surface_conditioning=1.2,
        ),
    }

    @classmethod
    def get(cls, name: str) -> Material:
        """Return a material by name."""

        key = name.lower()
        if key not in cls._materials:
            raise KeyError(f"Unknown material '{name}'")
        return cls._materials[key]
