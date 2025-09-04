"""Geometry utilities for DPF simulations."""

from .inductance import coaxial_inductance, loop_mutual_inductance
from .loaders import (
    load_axisymmetric_mesh,
    load_cad_geometry,
    load_unstructured_mesh,
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
)

__all__ = [
    "coaxial_inductance",
    "loop_mutual_inductance",
    "load_cad_geometry",
    "load_axisymmetric_mesh",
    "load_unstructured_mesh",
    "AxisymmetricProfile",
    "TaperedGeometry",
    "HollowGeometry",
    "ReentrantGeometry",
    "triple_junction_field",
    "set_triple_junction_field_map",
]
