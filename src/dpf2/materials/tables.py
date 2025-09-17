"""Lookup tables for basic material electrical properties.

The real project stores extensive tabulated data for many materials.  For the
purposes of the exercises we provide only a tiny subset used by the unit tests.
The values are intentionally simple and should not be interpreted as rigorous
engineering data.  They merely capture the order of magnitude of resistivity and
skin‑effect behaviour to exercise the surrounding code.
"""

from __future__ import annotations

from typing import Dict

# Resistivity in ohm metre at room temperature.  Only a couple of materials are
# required for the tests, additional entries can be added easily if needed.
RESISTIVITY_TABLE: Dict[str, float] = {
    "copper": 1.68e-8,
    "aluminum": 2.82e-8,
    "stainless_steel": 7.4e-7,
}

# Empirical coefficients for the simplistic skin effect model used by
# :class:`dpf2.circuit.distributed.TransmissionLineSegment`.  The coefficient is
# multiplied by ``sqrt(frequency)`` to obtain an additional per‑metre resistance
# contribution.  Units are ohm / metre / sqrt(Hz).
SKIN_EFFECT_TABLE: Dict[str, float] = {
    "copper": 6.0e-5,
    "aluminum": 7.5e-5,
    "stainless_steel": 1.2e-4,
}


def get_resistivity(material: str) -> float:
    """Return the resistivity for ``material``.

    ``material`` is matched case‑insensitively.  A :class:`KeyError` is raised if
    the material is unknown.
    """

    key = material.lower()
    if key not in RESISTIVITY_TABLE:
        raise KeyError(f"Unknown material '{material}'")
    return RESISTIVITY_TABLE[key]


def get_skin_effect_coeff(material: str) -> float:
    """Return the skin effect coefficient for ``material``.

    ``material`` is matched case‑insensitively.  A :class:`KeyError` is raised if
    the material is unknown.
    """

    key = material.lower()
    if key not in SKIN_EFFECT_TABLE:
        raise KeyError(f"Unknown material '{material}'")
    return SKIN_EFFECT_TABLE[key]


__all__ = [
    "RESISTIVITY_TABLE",
    "SKIN_EFFECT_TABLE",
    "get_resistivity",
    "get_skin_effect_coeff",
]
