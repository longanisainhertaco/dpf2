from __future__ import annotations

"""Utilities for geometry-dependent triple-junction field maps."""

from typing import Dict

# Default field map factors for common geometry presets.  These are
# normalized values used for simple testing; real simulations may replace
# them with calibrated data.
_TRIPLE_JUNCTION_FIELD_MAP: Dict[str, float] = {
    "mather": 1.0,
    "filippov": 0.9,
    "tapered": 1.1,
    "hollow": 1.05,
    "reentrant": 1.2,
}


def triple_junction_field(geometry: str, default: float = 1.0) -> float:
    """Return the triple-junction field factor for ``geometry``.

    Parameters
    ----------
    geometry:
        Name of the geometry preset.
    default:
        Value returned when ``geometry`` is unknown.
    """

    return _TRIPLE_JUNCTION_FIELD_MAP.get(geometry, default)


def set_triple_junction_field_map(geometry: str, field: float) -> None:
    """Override or define a field map entry for ``geometry``."""

    _TRIPLE_JUNCTION_FIELD_MAP[geometry] = field


def triple_junction_enhancement(
    geometry: str, anode_radius: float, cathode_radius: float, default: float = 1.0
) -> float:
    """Return a geometry-dependent triple-junction field enhancement.

    The enhancement scales the preset field factor returned by
    :func:`triple_junction_field` using the cathode to anode radius ratio.
    This simple relationship captures the intuition that larger cathode
    structures enhance the local electric field near the triple junction.

    Parameters
    ----------
    geometry:
        Name of the geometry preset.
    anode_radius, cathode_radius:
        Characteristic radii in centimetres.
    default:
        Baseline value used when ``geometry`` is unknown.
    """

    if anode_radius <= 0 or cathode_radius <= 0:
        raise ValueError("radii must be positive")
    base = triple_junction_field(geometry, default)
    ratio = cathode_radius / anode_radius
    return base * (1.0 + 0.1 * ratio)


__all__ = [
    "triple_junction_field",
    "set_triple_junction_field_map",
    "triple_junction_enhancement",
]
