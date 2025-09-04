from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .loaders import load_axisymmetric_mesh


@dataclass
class AxisymmetricProfile:
    """Axisymmetric mesh profile defined by radial and axial coordinates.

    Parameters
    ----------
    r: list of float
        Radial coordinate values.
    z: list of float
        Axial coordinate values.
    """

    r: List[float]
    z: List[float]

    @classmethod
    def from_file(cls, path: Path) -> "AxisymmetricProfile":
        """Create an :class:`AxisymmetricProfile` by loading ``path``.

        The file may be in the simple JSON/text formats accepted by
        :func:`dpf2.geometry.loaders.load_axisymmetric_mesh` or an STL/VTK
        surface mesh describing the profile.
        """

        data = load_axisymmetric_mesh(path)
        return cls(r=data["r"], z=data["z"])
