"""Minimal material property database with provenance.

The values are approximate and intended for unit tests.  Each entry
includes a short provenance note describing the data source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MaterialProperties:
    """Container for simple material properties."""

    resistivity_ohm_m: float
    see_yield: float
    work_function_eV: float
    source: str


MATERIAL_PROPERTIES: Dict[str, MaterialProperties] = {
    "copper": MaterialProperties(
        resistivity_ohm_m=1.68e-8,
        see_yield=1.0,
        work_function_eV=4.7,
        source=(
            "Matula, J. Phys. Chem. Ref. Data 8, 1147 (1979); "
            "Scholtz et al., J. Vac. Sci. Technol. A7, 123 (1989)"
        ),
    ),
    "tungsten": MaterialProperties(
        resistivity_ohm_m=5.6e-8,
        see_yield=1.3,
        work_function_eV=4.5,
        source=(
            "Matula, J. Phys. Chem. Ref. Data 8, 1147 (1979); "
            "Vaughan, IEEE Trans. Electron Devices 36, 1963 (1989)"
        ),
    ),
}


def get_material_properties(name: str) -> MaterialProperties:
    """Return properties for a material by name.

    Parameters
    ----------
    name:
        Material identifier.  The lookup is case-insensitive.
    """

    return MATERIAL_PROPERTIES[name.lower()]
