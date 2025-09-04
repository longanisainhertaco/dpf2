
"""Material-related models and helpers."""

from .models import MaterialRef

from .library import MaterialLibrary

from .state import ComponentMaterialState
from .mdm import MaterialDamageModel
from .sputtering import (
    Species,
    sigmund_yield,
    yamamura_yield,
    impurity_source_terms,
)
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
    "Species",
    "sigmund_yield",
    "yamamura_yield",
    "impurity_source_terms",
    "RESISTIVITY_TABLE",
    "SKIN_EFFECT_TABLE",
    "get_resistivity",
    "get_skin_effect_coeff",
]

