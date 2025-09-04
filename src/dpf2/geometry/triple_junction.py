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


__all__ = ["triple_junction_field", "set_triple_junction_field_map"]
