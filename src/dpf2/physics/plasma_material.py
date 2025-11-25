from __future__ import annotations

"""Lightweight plasma–material interaction helpers."""

from dataclasses import dataclass, field
from typing import Dict

from dpf2.materials.library import MaterialLibrary
from dpf2.materials.models import ImpurityState
from dpf2.materials.sputtering import Species, impurity_source_terms, yamamura_yield


@dataclass
class PlasmaMaterialInteraction:
    """Track surface erosion and impurity sources for simple couplings."""

    erosion_depths: Dict[str, float] = field(default_factory=dict)
    impurities: ImpurityState = field(default_factory=ImpurityState)

    def apply_flux(
        self,
        material_id: str,
        ion_flux: float,
        ion_species: Species,
        *,
        energy_eV: float = 500.0,
        angle_deg: float = 0.0,
        dt: float = 1e-6,
    ) -> Dict[str, float]:
        """Apply an incident ion flux to a surface and update inventories."""

        material = MaterialLibrary.get(material_id)
        target = Species(material.name, Z=int(material.atomic_mass // 2), mass_u=material.atomic_mass)
        yield_per_ion = yamamura_yield(ion_species, target, energy_eV, angle_deg)
        sources = impurity_source_terms(ion_flux, yield_per_ion, target)
        self.impurities.apply_sources(sources, dt)

        erosion = ion_flux * yield_per_ion * dt / max(material.density, 1.0)
        self.erosion_depths[material_id] = self.erosion_depths.get(material_id, 0.0) + erosion
        return sources


__all__ = ["PlasmaMaterialInteraction"]
