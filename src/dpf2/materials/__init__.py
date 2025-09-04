"""Material definition and lifecycle tracking utilities."""

from .library import Material, MaterialLibrary
from .state import ComponentMaterialState
from .mdm import MaterialDamageModel

__all__ = [
    "Material",
    "MaterialLibrary",
    "ComponentMaterialState",
    "MaterialDamageModel",
]
