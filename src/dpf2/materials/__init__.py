
"""Material-related models and helpers."""

from .models import MaterialRef

from .library import MaterialLibrary

from .state import ComponentMaterialState
from .mdm import MaterialDamageModel
from .tables import (
    RESISTIVITY_TABLE,
    SKIN_EFFECT_TABLE,
    get_resistivity,
    get_skin_effect_coeff,
)

__all__ = [
    "MaterialRef",

    "MaterialLibrary",
    "ComponentMaterialState",
    "MaterialDamageModel",
    "RESISTIVITY_TABLE",
    "SKIN_EFFECT_TABLE",
    "get_resistivity",
    "get_skin_effect_coeff",
]

