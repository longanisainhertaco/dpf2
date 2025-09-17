from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Dict

from ..utils import BaseModel, ConfigDict, Field

from ..core_schema import to_camel_case


class MaterialRef(BaseModel):
    """Reference to a material and optional coating information."""

    material_id: str = Field(..., alias="materialId")
    coating_id: Optional[str] = Field(None, alias="coatingId")

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )


@dataclass
class ImpurityState:
    """Track impurity surface densities produced by sputtering."""

    densities: Dict[str, float] = field(default_factory=dict)

    def apply_sources(self, sources: Dict[str, float], dt: float) -> None:
        """Update the impurity inventory with source fluxes.

        Parameters
        ----------
        sources:
            Mapping of species name to flux [m^-2 s^-1].
        dt:
            Time step over which the flux acts.
        """

        for sp, flux in sources.items():
            if flux <= 0:
                continue
            self.densities[sp] = self.densities.get(sp, 0.0) + flux * dt


__all__ = ["MaterialRef", "ImpurityState"]
