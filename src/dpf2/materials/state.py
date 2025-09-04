from dataclasses import dataclass, field
from typing import Dict, List

from .library import Material, MaterialLibrary


@dataclass
class ComponentMaterialState:
    """Runtime state for a component's material."""

    material: Material
    erosion: float = 0.0
    redeposited_mass: float = 0.0
    contamination_thickness: float = 0.0
    temperature_history: List[float] = field(default_factory=list)

    def record_temperature(self, temp: float) -> None:
        self.temperature_history.append(temp)

    def erode(self, amount: float) -> None:
        self.erosion += amount

    def redeposit(self, amount: float) -> None:
        self.redeposited_mass += amount

    def deposit(self, amount: float) -> None:
        self.contamination_thickness += amount

    def to_dict(self) -> Dict[str, object]:
        return {
            "material": self.material.name,
            "erosion": self.erosion,
            "redeposited_mass": self.redeposited_mass,
            "contamination_thickness": self.contamination_thickness,
            "temperature_history": list(self.temperature_history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ComponentMaterialState":
        mat = MaterialLibrary.get(str(data["material"]))
        return cls(
            material=mat,
            erosion=float(data.get("erosion", 0.0)),
            redeposited_mass=float(data.get("redeposited_mass", 0.0)),
            contamination_thickness=float(data.get("contamination_thickness", 0.0)),
            temperature_history=list(data.get("temperature_history", [])),
        )
