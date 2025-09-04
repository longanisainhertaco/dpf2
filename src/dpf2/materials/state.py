from dataclasses import dataclass, field
from typing import Dict, List

from .library import Material, MaterialLibrary


@dataclass
class ComponentMaterialState:
    """Runtime state for a component's material."""

    material: Material
    erosion: float = 0.0
    film_thickness: float = 0.0
    temperature_history: List[float] = field(default_factory=list)

    def record_temperature(self, temp: float) -> None:
        self.temperature_history.append(temp)

    def erode(self, amount: float) -> None:
        self.erosion += amount

    def deposit(self, amount: float) -> None:
        self.film_thickness += amount

    def to_dict(self) -> Dict[str, object]:
        return {
            "material": self.material.name,
            "erosion": self.erosion,
            "film_thickness": self.film_thickness,
            "temperature_history": list(self.temperature_history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ComponentMaterialState":
        mat = MaterialLibrary.get(str(data["material"]))
        return cls(
            material=mat,
            erosion=float(data.get("erosion", 0.0)),
            film_thickness=float(data.get("film_thickness", 0.0)),
            temperature_history=list(data.get("temperature_history", [])),
        )
