"""Geometry utilities for DPF simulations."""

from .inductance import coaxial_inductance, loop_mutual_inductance
from .loaders import (
    load_axisymmetric_mesh,
    load_cad_geometry,
    load_unstructured_mesh,
)
from .parameterized import (
    TaperedGeometry,
    HollowGeometry,
    ReentrantGeometry,
)

__all__ = [
    "coaxial_inductance",
    "loop_mutual_inductance",
    "load_cad_geometry",
    "load_axisymmetric_mesh",
    "load_unstructured_mesh",
    "TaperedGeometry",
    "HollowGeometry",
    "ReentrantGeometry",
]
