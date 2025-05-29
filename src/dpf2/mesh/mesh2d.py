"""2D cylindrical mesh utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class MeshCell:
    """Represents a single cell in the 2D mesh."""
    r_index: int
    z_index: int
    r_center: float
    z_center: float


class Mesh2D:
    """Simple 2D cylindrical mesh.

    Parameters
    ----------
    r_min, r_max : float
        Radial domain extent [m].
    z_min, z_max : float
        Axial domain extent [m].
    nr, nz : int
        Number of cells in the r and z directions.
    """

    def __init__(self, r_min: float, r_max: float, z_min: float, z_max: float, nr: int, nz: int) -> None:
        self.r = np.linspace(r_min, r_max, nr + 1)
        self.z = np.linspace(z_min, z_max, nz + 1)
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]
        self.nr = nr
        self.nz = nz
        self.cells: List[MeshCell] = self._create_cells()

    def _create_cells(self) -> List[MeshCell]:
        cells = []
        for i in range(self.nr):
            for j in range(self.nz):
                r_c = 0.5 * (self.r[i] + self.r[i + 1])
                z_c = 0.5 * (self.z[j] + self.z[j + 1])
                cells.append(MeshCell(i, j, r_c, z_c))
        return cells

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.nr and 0 <= nj < self.nz:
                neighbors.append((ni, nj))
        return neighbors
