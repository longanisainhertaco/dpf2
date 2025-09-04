"""Material interaction helpers coupling sputtering to impurity tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ...materials.sputtering import (
    Species,
    sigmund_yield,
    yamamura_yield,
    impurity_source_terms,
)

from .material_properties import MATERIAL_PROPERTIES, get_material_properties

__all__ = [
    "Species",
    "sigmund_yield",
    "yamamura_yield",
    "impurity_source_terms",
    "ImpurityState",
    "MATERIAL_PROPERTIES",
    "get_material_properties",
]


@dataclass
class ImpurityState:
    """Track impurity densities and compute effective charge."""

    densities: Dict[str, float] = field(default_factory=dict)

    def update(self, sources: Dict[str, float]) -> None:
        """Accumulate impurity source terms into the state."""

        for name, flux in sources.items():
            self.densities[name] = self.densities.get(name, 0.0) + flux

    def z_eff(self, charges: Dict[str, int]) -> float:
        """Return the effective charge ``Z_eff`` for current impurities."""

        num = 0.0
        den = 0.0
        for name, density in self.densities.items():
            Z = charges.get(name)
            if Z is None:
                continue
            num += density * (Z ** 2)
            den += density * Z
        return num / den if den > 0 else 0.0

    def feed_to_transport(self, transport_model) -> None:
        """Feed impurity densities into a transport model if supported."""

        if hasattr(transport_model, "set_impurities"):
            transport_model.set_impurities(self.densities)

    def feed_to_radiation(self, radiation_model, charges: Dict[str, int]) -> None:
        """Provide ``Z_eff`` to a radiation model if supported."""

        if hasattr(radiation_model, "set_z_eff"):
            radiation_model.set_z_eff(self.z_eff(charges))
