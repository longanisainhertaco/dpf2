"""3D Cartesian mesh utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
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

__all__ = ["Mesh3D", "MeshCell3D"]

