"""Boundary condition helpers for mesh-based field lists."""

from __future__ import annotations

from typing import Literal


def apply_bc(
    field: list,
    bc: Literal["periodic", "neumann", "dirichlet", "reflective", "absorbing"],
    axis: int,
    side: Literal["low", "high"],
    ghosts: int = 1,
) -> None:
    """Apply a basic boundary condition to ``field`` in-place.

    The ``field`` is expected to be a three-dimensional nested list with
    layout ``[x][y][z]`` including ghost cells on all sides.  Only the required
    subset of standard boundary types is supported and implemented using plain
    Python loops so that the helper works even when NumPy is unavailable.
    """

    def _copy_plane(src):
        return [row[:] for row in src]

    if axis == 0:
        for i in range(ghosts):
            src_low = field[-2 * ghosts + i]
            src_high = field[ghosts + i]
            if bc == "periodic":
                if side == "low":
                    field[i] = _copy_plane(src_low)
                else:
                    field[-ghosts + i] = _copy_plane(src_high)
            elif bc == "neumann":
                if side == "low":
                    field[i] = _copy_plane(field[ghosts + i])
                else:
                    field[-ghosts + i] = _copy_plane(field[-2 * ghosts + i])
            elif bc == "reflective":
                if side == "low":
                    field[i] = _copy_plane(field[2 * ghosts - 1 - i])
                else:
                    field[-ghosts + i] = _copy_plane(field[-ghosts - 1 - i])
            elif bc == "dirichlet":
                target = field[i] if side == "low" else field[-ghosts + i]
                for j in range(len(target)):
                    for k in range(len(target[j])):
                        target[j][k] = 0.0
            elif bc == "absorbing":
                target = field[i] if side == "low" else field[-ghosts + i]
                for j in range(len(target)):
                    for k in range(len(target[j])):
                        target[j][k] = 0.0
            else:  # pragma: no cover
                raise ValueError(bc)
    elif axis == 1:
        for j in range(ghosts):
            src_low = [col[-2 * ghosts + j] for col in field]
            src_high = [col[ghosts + j] for col in field]
            if bc == "periodic":
                if side == "low":
                    for i in range(len(field)):
                        field[i][j] = field[i][-2 * ghosts + j][:]
                else:
                    for i in range(len(field)):
                        field[i][-ghosts + j] = field[i][ghosts + j][:]
            elif bc == "neumann":
                if side == "low":
                    for i in range(len(field)):
                        field[i][j] = field[i][ghosts + j][:]
                else:
                    for i in range(len(field)):
                        field[i][-ghosts + j] = field[i][-2 * ghosts + j][:]
            elif bc == "reflective":
                if side == "low":
                    for i in range(len(field)):
                        field[i][j] = field[2 * ghosts - 1 - i][j][:]
                else:
                    for i in range(len(field)):
                        field[i][-ghosts + j] = field[-ghosts - 1 - i][j][:]
            elif bc == "dirichlet":
                for i in range(len(field)):
                    target = field[i][j] if side == "low" else field[i][-ghosts + j]
                    for k in range(len(target)):
                        target[k] = 0.0
            elif bc == "absorbing":
                for i in range(len(field)):
                    target = field[i][j] if side == "low" else field[i][-ghosts + j]
                    for k in range(len(target)):
                        target[k] = 0.0
            else:  # pragma: no cover
                raise ValueError(bc)
    elif axis == 2:
        for k in range(ghosts):
            if bc == "periodic":
                if side == "low":
                    for i in range(len(field)):
                        for j in range(len(field[i])):
                            field[i][j][k] = field[i][j][-2 * ghosts + k]
                else:
                    for i in range(len(field)):
                        for j in range(len(field[i])):
                            field[i][j][-ghosts + k] = field[i][j][ghosts + k]
            elif bc == "neumann":
                if side == "low":
                    for i in range(len(field)):
                        for j in range(len(field[i])):
                            field[i][j][k] = field[i][j][ghosts + k]
                else:
                    for i in range(len(field)):
                        for j in range(len(field[i])):
                            field[i][j][-ghosts + k] = field[i][j][-2 * ghosts + k]
            elif bc == "reflective":
                if side == "low":
                    for i in range(len(field)):
                        for j in range(len(field[i])):
                            field[i][j][k] = field[2 * ghosts - 1 - i][j][k]
                else:
                    for i in range(len(field)):
                        for j in range(len(field[i])):
                            field[i][j][-ghosts + k] = field[-ghosts - 1 - i][j][-ghosts + k]
            elif bc == "dirichlet":
                for i in range(len(field)):
                    for j in range(len(field[i])):
                        if side == "low":
                            field[i][j][k] = 0.0
                        else:
                            field[i][j][-ghosts + k] = 0.0
            elif bc == "absorbing":
                for i in range(len(field)):
                    for j in range(len(field[i])):
                        if side == "low":
                            field[i][j][k] = 0.0
                        else:
                            field[i][j][-ghosts + k] = 0.0
            else:  # pragma: no cover
                raise ValueError(bc)
    else:  # pragma: no cover
        raise ValueError(f"unknown axis {axis}")


__all__ = ["apply_bc"]

