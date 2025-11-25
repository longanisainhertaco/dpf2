from __future__ import annotations

"""Inductance calculations based on simple Biot–Savart integrations."""

import math

MU0 = 4e-7 * math.pi


def coaxial_inductance(inner_radius: float, outer_radius: float, length: float, *, n: int = 1000) -> float:
    """Return inductance of a straight coaxial plasma column."""

    dr = (outer_radius - inner_radius) / max(n - 1, 1)
    flux = 0.0
    r = inner_radius
    for _ in range(n):
        B = MU0 / (2 * math.pi * r)
        flux += B * dr
        r += dr
    return flux * length


def loop_mutual_inductance(r1: float, r2: float, separation: float) -> float:
    """Approximate mutual inductance between two coaxial circular loops.

    Parameters
    ----------
    r1, r2:
        Radii of the two loops in metres.
    separation:
        Axial separation between the loop centres in metres.
    """

    return MU0 * math.pi * r1 ** 2 * r2 ** 2 / (2 * (r1 ** 2 + separation ** 2) ** 1.5)


def reconstruct_dynamic_inductance(
    plasma_radii: list[float] | tuple[float, ...] | range | set | frozenset,
    *,
    length: float,
    sheath_thickness: float = 5e-3,
) -> list[float]:
    """Reconstruct a time-varying inductance from plasma radii.

    The helper treats each radius as a coaxial column with a thin return sheath
    and integrates the Biot–Savart contribution using
    :func:`coaxial_inductance`.  It intentionally uses minimal inputs so that
    callers can plug in radii extracted from parametric CAD features or from a
    puff/flashover timeline.
    """

    values: list[float] = []
    for r in plasma_radii:
        inner = max(r - sheath_thickness, 1e-6)
        outer = r + sheath_thickness
        values.append(coaxial_inductance(inner, outer, length))
    return values


__all__ = ["coaxial_inductance", "loop_mutual_inductance", "reconstruct_dynamic_inductance"]
