
"""Material-related models and helpers."""

from .models import MaterialRef
from .library import Material, MaterialLibrary
from .state import ComponentMaterialState
from .mdm import MaterialDamageModel

__all__ = [
    "MaterialRef",
    "Material",
    "MaterialLibrary",
    "ComponentMaterialState",
    "MaterialDamageModel",
]

