"""3D Cartesian mesh utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np


@dataclass
class MeshCell3D:
    """Represents a single cell in the 3D mesh."""
    x_index: int
    y_index: int
    z_index: int
    x_center: float
    y_center: float
    z_center: float


class Mesh3D:
    """Simple 3D Cartesian mesh.

    Parameters
    ----------
    x_min, x_max : float
        Domain extent in ``x`` [m].
    y_min, y_max : float
        Domain extent in ``y`` [m].
    z_min, z_max : float
        Domain extent in ``z`` [m].
    nx, ny, nz : int
        Number of cells in the ``x``, ``y`` and ``z`` directions.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        self.x = np.linspace(x_min, x_max, nx + 1)
        self.y = np.linspace(y_min, y_max, ny + 1)
        self.z = np.linspace(z_min, z_max, nz + 1)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.cells: List[MeshCell3D] = self._create_cells()

    # ------------------------------------------------------------------
    def _create_cells(self) -> List[MeshCell3D]:
        cells: List[MeshCell3D] = []
        for i in range(self.nx):
            x_c = 0.5 * (self.x[i] + self.x[i + 1])
            for j in range(self.ny):
                y_c = 0.5 * (self.y[j] + self.y[j + 1])
                for k in range(self.nz):
                    z_c = 0.5 * (self.z[k] + self.z[k + 1])
                    cells.append(MeshCell3D(i, j, k, x_c, y_c, z_c))
        return cells

    # ------------------------------------------------------------------
    def cell(self, i: int, j: int, k: int) -> MeshCell3D:
        """Return the :class:`MeshCell3D` at indices ``(i, j, k)``.

        The mesh is stored in ``x-major`` ordering (``i`` varies slowest and
        ``k`` fastest).  A simple index calculation retrieves the requested
        cell without constructing temporary arrays.
        """

        if not (0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz):
            raise IndexError("cell indices out of bounds")
        idx = i * self.ny * self.nz + j * self.nz + k
        return self.cells[idx]

    # ------------------------------------------------------------------
    def get_neighbors(self, i: int, j: int, k: int) -> List[Tuple[int, int, int]]:
        """Return indices of neighboring cells to ``(i, j, k)``."""
        neighbors: List[Tuple[int, int, int]] = []
        for di, dj, dk in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                neighbors.append((ni, nj, nk))
        return neighbors

    # ------------------------------------------------------------------
    def cell_volume(self) -> float:
        """Return the volume of a single cell."""
        return self.dx * self.dy * self.dz

    def face_areas(self) -> Tuple[float, float, float]:
        """Return the areas of cell faces normal to ``x``, ``y`` and ``z``."""
        return (
            self.dy * self.dz,
            self.dx * self.dz,
            self.dx * self.dy,
        )

    # ------------------------------------------------------------------
    def map_curved_boundary(self, func: Callable[[float, float], float]) -> np.ndarray:
        """Return coordinates of a curved ``z`` surface.

        The provided ``func`` is expected to take ``(x, y)`` coordinates and
        return the ``z`` location of the curved surface.  The function is
        evaluated at the cell centers of the top boundary and a two-dimensional
        array of surface locations is returned.  This utility is intentionally
        lightweight and does not attempt any advanced geometric processing; it
        merely offers a convenient way for tests or simple applications to map
        a planar mesh onto a curved physical boundary.
        """

        surface = np.zeros((self.nx, self.ny))
        for i in range(self.nx):
            x_c = 0.5 * (self.x[i] + self.x[i + 1])
            for j in range(self.ny):
                y_c = 0.5 * (self.y[j] + self.y[j + 1])
                surface[i, j] = func(x_c, y_c)
        return surface

    # ------------------------------------------------------------------
    def interpolate_ghost_cells(
        self,
        field: np.ndarray,
        axis: int,
        side: str,
        ghosts: int,
        surface: Callable[[float, float], float],
        value: Callable[[float, float, float], float],
    ) -> None:
        """Linearly interpolate ghost-cell values for a curved boundary.

        This helper currently only supports boundaries normal to the ``z`` axis.
        Passing ``axis`` as ``0`` or ``1`` will raise :class:`NotImplementedError`.

        Parameters
        ----------
        field:
            Numpy array with ghost cells on all sides.  The array is modified in
            place.
        axis:
            Direction normal to the boundary: ``0`` for ``x``, ``1`` for ``y``
            and ``2`` for ``z``.  Only ``axis=2`` is implemented.
        side:
            ``"low"`` or ``"high"`` indicating which side to operate on.
        ghosts:
            Number of ghost cells present in ``field``.
        surface, value:
            ``surface`` provides the physical location of the curved boundary
            while ``value`` supplies the desired field value at that location.

        Only a straightforward linear extrapolation is implemented which is
        sufficient for regression tests and simple applications.
        """

        arr = field
        if axis != 2:  # pragma: no cover - only ``z`` boundaries are supported
            raise NotImplementedError("only z-axis interpolation is implemented")

        k_int = ghosts if side == "low" else ghosts + self.nz - 1
        k_ghost = k_int - 1 if side == "low" else k_int + 1

        for i in range(self.nx):
            ii = ghosts + i
            x_c = 0.5 * (self.x[i] + self.x[i + 1])
            for j in range(self.ny):
                jj = ghosts + j
                y_c = 0.5 * (self.y[j] + self.y[j + 1])
                z_c = 0.5 * (self.z[k_int - ghosts] + self.z[k_int - ghosts + 1])
                boundary_z = surface(x_c, y_c)
                boundary_value = value(x_c, y_c, boundary_z)
                distance_surface = (
                    z_c - boundary_z if side == "low" else boundary_z - z_c
                )
                if distance_surface == 0:
                    arr[ii][jj][k_ghost] = boundary_value
                else:
                    interior = arr[ii][jj][k_int]
                    arr[ii][jj][k_ghost] = interior + (
                        (boundary_value - interior) * self.dz / distance_surface
                    )

__all__ = ["Mesh3D", "MeshCell3D"]

