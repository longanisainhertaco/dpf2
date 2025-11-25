"""Geometry utilities for DPF simulations."""

from .inductance import (
    coaxial_inductance,
    loop_mutual_inductance,
    reconstruct_dynamic_inductance,
)
from .loaders import (
    load_axisymmetric_mesh,
    load_cad_geometry,
    load_unstructured_mesh,
)
from .importer import (
    ImportedGeometry,
    load_geometry_with_materials,
    parametrized_geometry,
)
from .axisymmetric import AxisymmetricProfile
from .parameterized import (
    TaperedGeometry,
    HollowGeometry,
    ReentrantGeometry,
)
from .triple_junction import (
    triple_junction_field,
    set_triple_junction_field_map,
    triple_junction_enhancement,
)

__all__ = [
    "coaxial_inductance",
    "loop_mutual_inductance",
    "reconstruct_dynamic_inductance",
    "load_cad_geometry",
    "load_axisymmetric_mesh",
    "load_unstructured_mesh",
    "ImportedGeometry",
    "load_geometry_with_materials",
    "parametrized_geometry",
    "AxisymmetricProfile",
    "TaperedGeometry",
    "HollowGeometry",
    "ReentrantGeometry",
    "triple_junction_field",
    "set_triple_junction_field_map",
    "triple_junction_enhancement",
]
