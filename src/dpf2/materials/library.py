from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Material:
    """Static material properties."""

    name: str
    density: float  # kg/m^3
    atomic_mass: float  # atomic mass units
    sputter_yield: float  # atoms removed per incident particle


class MaterialLibrary:
    """Simple registry of materials with basic properties."""

    _materials: Dict[str, Material] = {
        "copper": Material("copper", density=8960.0, atomic_mass=63.546, sputter_yield=0.01),
        "tungsten": Material("tungsten", density=19300.0, atomic_mass=183.84, sputter_yield=0.005),
        "stainless_steel": Material("stainless_steel", density=8000.0, atomic_mass=55.845, sputter_yield=0.02),
    }

    @classmethod
    def get(cls, name: str) -> Material:
        """Return a material by name."""

        key = name.lower()
        if key not in cls._materials:
            raise KeyError(f"Unknown material '{name}'")
        return cls._materials[key]
